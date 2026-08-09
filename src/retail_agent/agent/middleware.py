"""Where the graph's edges went.

The graph's claim was that every safety property was an edge rather than an
instruction in a prompt. That claim survives here — the properties are
middleware and tool preconditions rather than prose — with one honest
difference: an edge is visible in a diagram and a middleware stack is a list.
Which is why this file is one list, built in one place, rather than options
scattered across call sites.

Two stacks, because the two agents bound different things. The supervisor
carries the human gate, the persona and the recording; the analyst carries the
query budget and the repair path.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
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
from langchain_core.messages import AIMessage

from retail_agent.agent.capture import PendingDefinition, TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import (
    CONVERSATION_SUMMARY_PROMPT,
    PERSONA_DEFAULT,
    SAFETY_RULES,
    SUPERVISOR_PROMPT,
)
from retail_agent.agent.reports import render_manifest, resolve_delete
from retail_agent.agent.tools import GuardRejection
from retail_agent.datasources.base import DataSourceError
from retail_agent.llm.messages import message_text
from retail_agent.llm.resilience import (
    BASE_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
    is_retryable,
)
from retail_agent.store.definitions import all_definitions
from retail_agent.store.personas import active_body
from retail_agent.store.preferences import notes_for, preference_block

log = logging.getLogger(__name__)

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
    deps: AgentDeps, capture: TurnCapture, *, pause_for_definitions: bool = False
) -> list[AgentMiddleware]:
    """The stack that bounds the turn.

    `pause_for_definitions` arms the pause on the `ask_for_definitions` tool.
    Off by default because a pause needs somebody there to answer it:
    `seams.ask_once` scores a paused turn as unanswered, and the eval cases that
    turn on an undefined term are the brief's own examples. Unarmed, the tool
    still runs — it finds nothing in the store, records the assumption and tells
    the model to choose and disclose.
    """
    stack: list[AgentMiddleware] = [_prompt(deps, capture), *_pii()]

    # After `_pii()`, and that placement is the guarantee rather than a
    # preference. Both hook `before_model` and those run in list order; the
    # summariser writes what it produces back into state, so running it first
    # would read unredacted warehouse output and persist it — turning a
    # transient exposure into a stored one.
    summarizer = _summarization(deps)
    if summarizer is not None:
        stack.append(summarizer)

    stack += [
        _approval_gate(deps, capture, pause_for_definitions=pause_for_definitions),
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


def supervisor_system_prompt(deps: AgentDeps, capture: TurnCapture) -> str:
    """The supervisor's system prompt for one model call.

    A plain function rather than only the closure below, because this is the
    part with a decision in it — what gets appended and when — and a test of it
    should not have to build a `ModelRequest` to ask.
    """
    prompt = SUPERVISOR_PROMPT.format(
        persona=active_body(deps.personas) or PERSONA_DEFAULT,
        safety=SAFETY_RULES,
    )
    # Appended only when there is something to append: a user with no notes
    # should get the same prompt as before this feature existed, not the same
    # prompt with two blank lines welded to the end.
    block = preference_block(notes_for(deps.preferences, capture.user_id))
    return f"{prompt}\n\n{block}" if block else prompt


def _prompt(deps: AgentDeps, capture: TurnCapture) -> AgentMiddleware:
    """The supervisor's system prompt, assembled per model call.

    Per call rather than at build time, and that is not a detail. The persona is
    a database row the CEO can edit weekly, and the preferences are rows the
    user can change mid-session; binding either when the agent is constructed
    means a long-lived process serves the old one until it restarts.
    """

    @dynamic_prompt
    def supervisor_prompt(request) -> str:
        return supervisor_system_prompt(deps, capture)

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


def _approval_gate(
    deps: AgentDeps, capture: TurnCapture, *, pause_for_definitions: bool = False
) -> AgentMiddleware:
    """The two places a turn stops for a person, as interrupts before the tool.

    Both `when` predicates resolve read-only, without a model call, and park
    what they found on the capture. That is what makes each one a gate rather
    than advice: a delete that matches nothing never raises a prompt, a term
    this executive has already defined never raises one either, and in both
    cases what the user is shown is exactly what the predicate found.

    What changed with `ask_for_definitions` is which half is deterministic.
    *Whether* a word needs settling is now the model's judgement, made by
    calling the tool at all. What survives here is the narrower check that does
    not need one: whether the answer is already on file.
    """

    def has_targets(request) -> bool:
        args = request.tool_call.get("args", {})
        capture.pending = resolve_delete(
            deps,
            capture,
            term=args.get("term", "") or "",
            session_scoped=bool(args.get("session_scoped", False)),
        )
        return capture.pending is not None

    def describe(tool_call, state, runtime) -> str:
        return (
            render_manifest(capture.pending)
            if capture.pending
            else "Delete saved reports?"
        )

    def still_unsettled(request) -> bool:
        # Only pause if the answer can be kept. Without a store the agent would
        # ask the same person the same question every turn, which is worse than
        # assuming and saying so — the bargain the tool's own body makes.
        if deps.definitions is None:
            return False

        args = request.tool_call.get("args", {})
        capture.pending_definition = open_terms(
            deps, capture, args.get("terms") or []
        )
        return capture.pending_definition is not None

    def describe_definition(tool_call, state, runtime) -> str:
        pending = capture.pending_definition
        return describe_open_terms(pending.terms) if pending else "A term needs defining."

    interrupt_on: dict[str, InterruptOnConfig] = {
        "delete_reports": InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            description=describe,
            when=has_targets,
        )
    }
    if pause_for_definitions:
        interrupt_on["ask_for_definitions"] = InterruptOnConfig(
            # `approve` runs the tool body, which reads the answers back out of
            # the store the CLI just wrote them to. "Decide for me" and "cancel"
            # are both `reject` with different messages: the body never runs, so
            # nothing has to rewrite the call's arguments to mean either.
            allowed_decisions=["approve", "reject"],
            description=describe_definition,
            when=still_unsettled,
        )

    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on, description_prefix="")


def open_terms(
    deps: AgentDeps, capture: TurnCapture, terms: Sequence[str]
) -> PendingDefinition | None:
    """Of the words the model asked about, the ones still worth a question.

    The model decides what it does not understand; this decides what has
    already been answered. Both matter, and only the second can be settled
    without a model: a term this executive defined last week must not produce
    the same prompt again, however reasonable the model was to ask.

    Never raises — `all_definitions` returns what it managed to read, so an
    unreachable store costs a question that did not need asking, never the turn.
    """
    known = all_definitions(deps.definitions, capture.user_id)
    still_open = tuple(
        term.strip()
        for term in terms
        if term and term.strip() and term.strip().lower() not in known
    )
    return PendingDefinition(terms=still_open) if still_open else None


def describe_open_terms(terms: Sequence[str]) -> str:
    """What the pause is about.

    The term alone. There is no gloss to add any more: the words are the
    executive's own rather than keys of a dict that shipped a description with
    each one, and `propose` puts concrete candidate meanings under this anyway.
    """
    return "\n".join(f"{term!r} has no agreed definition yet" for term in terms)


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
        message = _last_ai_message(state)
        answer = message_text(message).strip() if message is not None else ""

        try:
            deps.traces.record(capture.to_trace(answer))
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
