"""Where the graph's edges went.

The graph's claim was that every safety property was an edge rather than an
instruction in a prompt. That claim survives here — the properties are
middleware and tool preconditions rather than prose — with one honest
difference: an edge is visible in a diagram and a middleware stack is a list.
Which is why this file is one list, built in one place, rather than options
scattered across call sites.

Two stacks, because the two agents bound different things. The supervisor
carries the persona and the recording; the analyst carries the query budget
and the repair path. The human gates used to live here too, as a
`HumanInTheLoopMiddleware` `when` predicate deciding whether to pause. They
don't any more: a `when` predicate is read-only and cannot touch the store or
write state, so it could resolve a pause but never answer one. Both gates
moved into the tool bodies that need the answers — `delete_reports` and
`ask_for_definitions` call `interrupt()` themselves, resolve what they showed
the user again on the way back, and act on the resume value directly.
"""

from __future__ import annotations

import logging
import time

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    after_agent,
    dynamic_prompt,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import (
    CONVERSATION_SUMMARY_PROMPT,
    PERSONA_DEFAULT,
    SAFETY_RULES,
    SUPERVISOR_PROMPT,
)
from retail_agent.agent.state import attempt_record, step_event
from retail_agent.agent.tools import GuardRejection
from retail_agent.datasources.base import DataSourceError
from retail_agent.llm.messages import message_text
from retail_agent.llm.resilience import (
    BASE_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
    is_retryable,
)
from retail_agent.obs.traces import trace_from_state
from retail_agent.store.personas import active_body
from retail_agent.store.preferences import notes_for, preference_block

log = logging.getLogger(__name__)


class MissingTurnIdentity(RuntimeError):
    """A turn started with no caller identity to act as.

    Every identity-scoped tool (`list_reports`, `remember_definition`,
    `report_writer`, `delete_reports`, and the rest) now reads
    `runtime.context.user_id` rather than a value closed over when the graph
    was built. `TurnContext`'s fields default to `""`, and on the LangGraph
    server a run posted with no `context` in its body is coerced to
    `TurnContext()` rather than left `None` (`create_valid_run` turns a
    missing context into `{}`, and `_coerce_context` turns `{}` into the
    dataclass's defaults). Nothing about that raises — every tool would
    simply read and write against the empty-string user, a shared bucket for
    every caller who forgot. That is worse than a crash, so this is the
    crash: raised before every tool call this turn makes — the first one and
    any a resume replays — instead of a silent misattribution discovered
    later. See `_IdentityGuardMiddleware`.
    """

# Redacted rather than blocked: blocking raises and kills the turn, and a turn
# that dies is an agent failure rather than the leak it was. This is the second
# line anyway — `mask_dataframe` stops PII entering context at all, and the SQL
# guard drops restricted columns so they never leave the warehouse.
# `apply_to_tool_results` is the setting that matters: PII arrives from the
# warehouse inside a ToolMessage, never from the executive's question.
PII_TYPES = ("email", "credit_card", "ip")

# A model that has stopped making progress still costs money on every turn.
MAX_MODEL_CALLS = 30


def analyst_middleware(deps: AgentDeps) -> list[AgentMiddleware]:
    """The stack that bounds the SQL loop.

    `run_limit` is computed from the same settings the budgets were computed
    from, rather than written down, so tuning one in config moves the agent with
    it. A loop silently allowed twice the queries would look like better
    accuracy for a reason no report would mention.

    Takes the whole `deps` rather than just `settings` because the provider
    chain reaches the model through here now: the fallbacks are built once at
    startup and carried on `deps`, not rebuilt per turn.
    """
    settings = deps.settings
    sql_budget = (
        settings.max_analysis_steps + settings.repair_budget + settings.diagnose_budget
    )

    return [
        *_pii(),
        ToolCallLimitMiddleware(tool_name="run_sql", run_limit=sql_budget),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        _SqlFailureRecorder(),
        *_resilience(deps),
    ]


def supervisor_middleware(deps: AgentDeps) -> list[AgentMiddleware]:
    """The stack that bounds the turn."""
    stack: list[AgentMiddleware] = [
        _identity_guard(),
        _prompt(deps),
        *_pii(),
    ]

    # After `_pii()`, and that placement is the guarantee rather than a
    # preference. Both hook `before_model` and those run in list order; the
    # summariser writes what it produces back into state, so running it first
    # would read unredacted warehouse output and persist it — turning a
    # transient exposure into a stored one.
    summarizer = _summarization(deps)
    if summarizer is not None:
        stack.append(summarizer)

    stack += [
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ToolErrorMiddleware(on_error=describe_failure),
        _recorder(deps),
        *_resilience(deps),
    ]
    return stack


def _resilience(deps: AgentDeps) -> list[AgentMiddleware]:
    """Retry the current provider, then fall over to the next one.

    Last in the stack, which makes it innermost: the composed handlers run
    first-is-outermost, so everything above counts one logical model call while
    this layer quietly spends several attempts on it. Put the other way round,
    `ModelCallLimitMiddleware` would bill a rate-limited turn three times.

    Order within the pair matters for the same reason. Fallback is outermost,
    so retries are exhausted against one provider before moving to the next —
    reversed, a single retry would restart the whole sweep.

    These replace a hand-written chain object that impersonated a chat model.
    Nothing here knows how to call a model; it is handed one.
    """
    stack: list[AgentMiddleware] = []

    if deps.llm_fallbacks:
        stack.append(ModelFallbackMiddleware(*deps.llm_fallbacks))

    stack.append(
        ModelRetryMiddleware(
            # `max_retries` counts attempts *after* the first, so a setting of
            # 3 total attempts is 2 retries.
            max_retries=max(0, deps.settings.llm_retry_attempts - 1),
            # Without this the default retries every exception, spending the
            # whole budget on a rejected API key before reaching a provider
            # that would have answered.
            retry_on=is_retryable,
            # The default is "continue", which swallows the exception and hands
            # the agent an AIMessage describing the failure. That would look
            # like an answer, and `ModelFallbackMiddleware` outside it would
            # never see a failure to fall back from.
            on_failure="error",
            initial_delay=BASE_DELAY_SECONDS,
            backoff_factor=2.0,
            max_delay=MAX_DELAY_SECONDS,
        )
    )
    return stack


def _pii() -> list[AgentMiddleware]:
    return [
        PIIMiddleware(
            pii_type,
            strategy="redact",
            apply_to_input=False,
            apply_to_tool_results=True,
            apply_to_output=False,
        )
        for pii_type in PII_TYPES
    ]


def describe_failure(error: Exception, request: object) -> str | None:
    """Turn a tool failure into the message the model gets back.

    Only recoverable failures are handled; anything else propagates, so an
    internal bug surfaces as a failed turn rather than as an agent that quietly
    worked around it.

    `request` is unused but not optional — `OnError` takes two arguments, and a
    one-argument version passes every test that does not call it, then raises
    TypeError against the live provider.

    Used two ways now: by `ToolErrorMiddleware` in `supervisor_middleware`'s
    stack (for whatever future supervisor tool might raise one of these), and
    called directly by `_SqlFailureRecorder` below, which needs the same text
    but has to attach it to a `Command` rather than a bare `ToolMessage`.
    """
    if isinstance(error, GuardRejection):
        return f"The query was rejected before running: {error}. Rewrite it."
    if isinstance(error, DataSourceError):
        return f"The query failed: {error}. Fix it and try again."
    return None


class _SqlFailureRecorder(AgentMiddleware):
    """Turns a rejected or failed `run_sql` call into the turn's record.

    `run_sql` still raises `GuardRejection`/`DataSourceError` exactly as it
    always has (see their docstrings) — a tool cannot both raise, which is
    what `describe_failure` needs to explain the failure, and return the
    `Command` that writes an attempt into `TurnState`, which is what a raise
    never does. This sits between the two: it lets the tool run, catches
    exactly the two exception types `describe_failure` already knows how to
    turn into a repair instruction, and converts each into the same shape a
    successful call returns — a `Command` carrying the repair `ToolMessage`,
    the attempt, the event and the `calls` increment — instead of an error
    that reaches nowhere. Anything else propagates unrecorded, same as
    before this existed: an internal bug should still kill the turn rather
    than read as an agent that quietly recovered from it.

    `GuardRejection` and `DataSourceError` are exactly the two types
    `test_only_run_sql_reads_the_warehouse` guarantees only `run_sql` can
    raise, so this needs no tool-name filter to stay scoped to it — and it
    replaces `ToolErrorMiddleware(on_error=describe_failure)` in
    `analyst_middleware`'s stack rather than sitting alongside it, since
    nothing else there can raise a type `describe_failure` recognises.
    """

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # No `except GraphBubbleUp: raise` here, unlike `ToolErrorMiddleware`.
        # There it is load-bearing — that middleware catches bare `Exception`,
        # which a `GraphBubbleUp` control-flow signal (an interrupt, a parent
        # `Command`) would otherwise be swallowed by. This `except` only ever
        # catches `GuardRejection`/`DataSourceError`, and `GraphBubbleUp` is
        # not a subclass of either, so it was never reachable here and would
        # have read as a safety net this middleware does not need.
        started = time.perf_counter()
        try:
            return handler(request)
        except (GuardRejection, DataSourceError) as exc:
            return _sql_failure_command(request, exc, started)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        started = time.perf_counter()
        try:
            return await handler(request)
        except (GuardRejection, DataSourceError) as exc:
            return _sql_failure_command(request, exc, started)


def _sql_failure_command(
    request: ToolCallRequest, exc: Exception, started: float
) -> Command:
    """The `Command` a failed or rejected `run_sql` call should have
    returned, had a raise not been the only way to reach `describe_failure`.

    `sql` comes off the tool call itself — the model's own draft — since
    `run_sql` had nothing else to attach it to before raising. `executed_sql`
    and `violations` come off the exception: `run_sql` set them at the point
    it raised, because only it knows the qualified query or the guard's
    verdict. `index` reads `request.state`, the analyst subgraph's own
    running `attempts` — the same count `run_sql`'s success path numbers
    against — so a rejected first query and a successful second one still
    read `q1`, `q2` rather than both claiming `q1`.

    Hardcodes `step_event("run_sql", ...)` and reads `args["sql"]` because
    `GuardRejection`/`DataSourceError` are, today, exactly the two types only
    `run_sql` can raise (`test_only_run_sql_reads_the_warehouse`). A future
    analyst tool that raised either would file a phantom `run_sql` attempt
    with `sql=""` — fine to widen with a tool-name check if that ever
    happens, not needed while the invariant holds.
    """
    text = describe_failure(exc, request) or str(exc)
    state = request.state or {}
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=text,
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            ],
            "attempts": [
                attempt_record(
                    sql=request.tool_call["args"].get("sql", ""),
                    executed_sql=getattr(exc, "executed_sql", None),
                    violations=getattr(exc, "violations", None),
                    error=None if isinstance(exc, GuardRejection) else str(exc),
                    index=len(state.get("attempts", [])),
                )
            ],
            "events": [step_event("run_sql", started, text)],
            "calls": 1,
        }
    )


def supervisor_system_prompt(deps: AgentDeps, *, user_id: str) -> str:
    """The supervisor's system prompt for one model call.

    A plain function rather than only the closure below, because this is the
    part with a decision in it — what gets appended and when — and a test of it
    should not have to build a `ModelRequest` to ask.

    Takes `user_id` as a plain parameter rather than reading it off anything
    stateful: identity lives on `TurnContext`, and this is not a tool, so
    `ToolRuntime` never reaches it. `_prompt`'s closure is what has a
    `runtime` to read it from.
    """
    prompt = SUPERVISOR_PROMPT.format(
        persona=active_body(deps.personas) or PERSONA_DEFAULT,
        safety=SAFETY_RULES,
    )
    # Appended only when there is something to append: a user with no notes
    # should get the same prompt as before this feature existed, not the same
    # prompt with two blank lines welded to the end.
    block = preference_block(notes_for(deps.preferences, user_id))
    return f"{prompt}\n\n{block}" if block else prompt


def _prompt(deps: AgentDeps) -> AgentMiddleware:
    """The supervisor's system prompt, assembled per model call.

    Per call rather than at build time, and that is not a detail. The persona is
    a database row the CEO can edit weekly, and the preferences are rows the
    user can change mid-session; binding either when the agent is constructed
    means a long-lived process serves the old one until it restarts.

    Also where a turn with no caller identity is refused before a model call
    is even made, for a turn that never reaches for a tool at all — a plain
    "hello" needs no `run_sql`, `list_reports` or anything else
    `_IdentityGuardMiddleware` wraps, so a turn like that would otherwise sail
    straight through on an empty-string user, all the way to a final answer.
    `_identity_guard`'s `wrap_tool_call` is what holds on every resume — this
    is a `before_model`-style hook, so it does not run again once a paused
    tool's resume re-enters the tools node directly — but it is the only
    thing that ever sees a turn with no tool call in it at all.
    """

    @dynamic_prompt
    def supervisor_prompt(request) -> str:
        _require_identity(request.runtime)
        return supervisor_system_prompt(deps, user_id=request.runtime.context.user_id)

    return supervisor_prompt


def _summarization(deps: AgentDeps) -> AgentMiddleware | None:
    """Compaction for a session that outgrows the model's window.

    Returns None when disabled rather than a middleware that never fires: a
    stack is the only place this agent's control flow is visible, and something
    listed there that cannot act makes it a worse description of the turn.

    Narrative only — `CONVERSATION_SUMMARY_PROMPT` forbids restating figures.
    A summariser that retypes "$412,880" produces, on the next turn, a number
    the model believes came from a query, which is precisely what `SAFETY_RULES`
    exists to prevent. Re-running the query is cheaper than reporting a wrong
    number.
    """
    trigger = deps.settings.context_summarize_trigger_tokens
    if trigger <= 0:
        return None

    return SummarizationMiddleware(
        model=deps.llm,
        trigger=("tokens", trigger),
        keep=("messages", deps.settings.context_keep_messages),
        summary_prompt=CONVERSATION_SUMMARY_PROMPT,
    )


class _IdentityGuardMiddleware(AgentMiddleware):
    """Refuse to run any tool body for a turn with no caller identity.

    This is the guard that actually holds on resume, and the reason it is a
    `wrap_tool_call` hook rather than a second `before_agent` check: LangGraph
    checkpoints `before_agent` as a completed node, so `Command(resume=...)`
    — which re-runs only the pending task, the tools node — never fires
    `before_agent` again. `wrap_tool_call` is not a node; it is baked into the
    tools node's own execution (`ToolNode(wrap_tool_call=...)` in
    `langchain.agents.factory`), so it runs on every tool call the tools node
    ever makes, including the replayed one a resume produces. `context` is
    supplied per `invoke`/`ainvoke` call and is not checkpointed, so a resume
    that omits it — exactly the shape the LangGraph server sends when a run
    was posted with no `context` in its body — must be caught here, before
    the tool body (in particular a paused gate's replayed body, which runs
    again on the way back from `interrupt()`) reads or writes against an
    empty-string user.

    Listed first in the stack so it is outermost among the middleware that
    wrap tool calls: it must run — and raise — before `ToolErrorMiddleware`
    or anything else gets a chance to touch the call.
    """

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        _require_identity(request.runtime)
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        _require_identity(request.runtime)
        return await handler(request)


def _identity_guard() -> AgentMiddleware:
    return _IdentityGuardMiddleware()


def _require_identity(runtime) -> None:
    """Fail the turn loudly, at the start, rather than let every tool below
    silently read and write against the empty-string user.

    See `MissingTurnIdentity` for why a run can reach this point with
    `runtime.context` looking populated but empty rather than `None`.
    """
    context = runtime.context
    if context is None or not getattr(context, "user_id", ""):
        raise MissingTurnIdentity(
            "This turn has no caller identity, so it was refused before any "
            "tool ran. Every tool that reads or writes report, definition or "
            "preference data takes the executive from runtime.context.user_id "
            "now, not from how the agent was built — without one they would "
            "all silently share an empty-string user.\n"
            "  In LangGraph Studio: set user_id in the run's configurable "
            "panel (session_id and turn_id are optional).\n"
            "  Over the API: post "
            '{"config": {"configurable": {"user_id": "..."}}} — supplying '
            "both configurable and a top-level context in one request is a "
            "400.\n"
            "  From Python: pass context=TurnContext(user_id=..., "
            "session_id=..., turn_id=...) to agent.invoke()/.ainvoke()."
        )


def _recorder(deps: AgentDeps) -> AgentMiddleware:
    """The trace, written on every path out of the turn.

    No output sweep here any more. Personal data is stopped where it enters
    rather than where it leaves: the SQL guard drops restricted columns,
    `mask_dataframe` masks what survives, and `_pii()` redacts the tool result
    before the model reads it. A final-answer regex could only catch what those
    three missed, and what reaches this hook is the text the user saw.
    """

    @after_agent
    def record(state, runtime) -> dict:
        # What this turn will cost every later turn: the thread is re-sent
        # whole on each model call. Approximate on purpose — an exact count is
        # a provider round trip per turn, spent to tune one setting.
        #
        # Written back into state (rather than only used locally) so a caller
        # reading the turn's final state — the CLI, the eval — sees the same
        # figure the trace was built from, and so a turn that dies before this
        # hook ever runs still reads as "not measured" rather than stale.
        context_tokens = count_tokens_approximately(state.get("messages", []) or [])

        message = _last_ai_message(state)
        answer = message_text(message).strip() if message is not None else ""

        context = runtime.context
        try:
            deps.traces.record(
                trace_from_state(
                    {**state, "context_tokens": context_tokens},
                    answer,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                )
            )
        except Exception as err:
            # Losing a trace should not lose the answer it describes.
            log.warning("could not record the trace (%s)", err)

        return {"context_tokens": context_tokens}

    return record


def _last_ai_message(state: dict):
    for message in reversed(state.get("messages", []) or []):
        if isinstance(message, AIMessage) and message_text(message).strip():
            return message
    return None
