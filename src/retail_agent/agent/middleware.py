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
    before_agent,
    dynamic_prompt,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import (
    CONVERSATION_SUMMARY_PROMPT,
    PERSONA_DEFAULT,
    SAFETY_RULES,
    SUPERVISOR_PROMPT,
)
from retail_agent.agent.tools import GuardRejection
from retail_agent.datasources.base import DataSourceError
from retail_agent.llm.messages import message_text
from retail_agent.llm.resilience import (
    BASE_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
    is_retryable,
)
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
        ToolErrorMiddleware(on_error=describe_failure),
        *_resilience(deps),
    ]


def supervisor_middleware(
    deps: AgentDeps, capture: TurnCapture
) -> list[AgentMiddleware]:
    """The stack that bounds the turn."""
    stack: list[AgentMiddleware] = [
        _identity_guard(),
        _turn_sync(capture),
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
        _recorder(deps, capture),
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
    """
    if isinstance(error, GuardRejection):
        return f"The query was rejected before running: {error}. Rewrite it."
    if isinstance(error, DataSourceError):
        return f"The query failed: {error}. Fix it and try again."
    return None


def supervisor_system_prompt(deps: AgentDeps, *, user_id: str) -> str:
    """The supervisor's system prompt for one model call.

    A plain function rather than only the closure below, because this is the
    part with a decision in it — what gets appended and when — and a test of it
    should not have to build a `ModelRequest` to ask.

    Takes `user_id` rather than a `TurnCapture`: identity lives on
    `TurnContext` now, and this is not a tool, so `ToolRuntime` never reaches
    it. `_prompt`'s closure is what has a `runtime` to read it from.
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
    """

    @dynamic_prompt
    def supervisor_prompt(request) -> str:
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


def _turn_sync(capture: TurnCapture) -> AgentMiddleware:
    """Point a long-lived capture at the turn actually being run.

    The CLI and the eval build a fresh capture per turn, question already set,
    and this hook changes nothing there. Studio compiles the agent once and
    closes its tools over one capture — so without this, `settled_meanings`
    retrieves for the build-time question ("") and caches the nothing it found,
    for the life of the process.

    The mid-turn cache is dropped only when the question changed: a resume
    after the definition pause is the same question, and `settled_meanings`'
    retrieval must survive it for the replayed tool body to read.

    This also runs `_require_identity` on the initial call, as a fail-fast: a
    turn started with no identity is rejected before a model call is even
    made, rather than after paying for one. That is a cost optimisation only
    — `before_agent` is a completed checkpoint node and never runs again on
    `Command(resume=...)`, so it cannot be what makes the guarantee hold. The
    guarantee — on the initial call, on every resume, before every tool —
    belongs to `_identity_guard` (`wrap_tool_call`) above, which is why that
    one is listed first in `supervisor_middleware` regardless of what this
    hook does.
    """

    @before_agent
    def sync(state, runtime) -> None:
        _require_identity(runtime)

        for message in reversed(state.get("messages", []) or []):
            if isinstance(message, HumanMessage):
                question = message_text(message).strip()
                if question and question != capture.question:
                    capture.question = question
                    capture.recalled_trios = None
                return None
        return None

    return sync


def _require_identity(runtime) -> None:
    """Fail the turn loudly, at the start, rather than let every tool below
    silently read and write against the empty-string user.

    See `MissingTurnIdentity` for why a run can reach this point with
    `runtime.context` looking populated but empty rather than `None`.
    """
    context = runtime.context
    if context is None or not getattr(context, "user_id", ""):
        raise MissingTurnIdentity(
            "This turn has no caller identity. Pass "
            "context=TurnContext(user_id=..., session_id=..., turn_id=...) "
            "to agent.invoke()/.ainvoke() — every tool that reads or writes "
            "report, definition or preference data reads it from "
            "runtime.context.user_id now, not from how the agent was built."
        )


def _recorder(deps: AgentDeps, capture: TurnCapture) -> AgentMiddleware:
    """The trace, written on every path out of the turn.

    No output sweep here any more. Personal data is stopped where it enters
    rather than where it leaves: the SQL guard drops restricted columns,
    `mask_dataframe` masks what survives, and `_pii()` redacts the tool result
    before the model reads it. A final-answer regex could only catch what those
    three missed, and what reaches this hook is the text the user saw.
    """

    @after_agent
    def record(state, runtime) -> None:
        # What this turn will cost every later turn: the thread is re-sent
        # whole on each model call. Approximate on purpose — an exact count is
        # a provider round trip per turn, spent to tune one setting.
        capture.context_tokens = count_tokens_approximately(
            state.get("messages", []) or []
        )

        message = _last_ai_message(state)
        answer = message_text(message).strip() if message is not None else ""

        context = runtime.context
        try:
            deps.traces.record(
                capture.to_trace(
                    answer,
                    user_id=context.user_id,
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                )
            )
        except Exception as err:
            # Losing a trace should not lose the answer it describes.
            log.warning("could not record the trace (%s)", err)

        return None

    return record


def _last_ai_message(state: dict):
    for message in reversed(state.get("messages", []) or []):
        if isinstance(message, AIMessage) and message_text(message).strip():
            return message
    return None
