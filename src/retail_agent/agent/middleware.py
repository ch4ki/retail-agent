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
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    after_agent,
    dynamic_prompt,
)
from langchain_core.messages import AIMessage

from retail_agent.agent.capture import PendingDefinition, TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import (
    PERSONA_DEFAULT,
    SAFETY_RULES,
    SUPERVISOR_PROMPT,
)
from retail_agent.agent.reports import render_manifest, resolve_delete
from retail_agent.agent.tools import GuardRejection, recall
from retail_agent.config import Settings
from retail_agent.datasources.base import DataSourceError
from retail_agent.knowledge.trios import UNDEFINED_TERMS, unresolved
from retail_agent.llm.messages import message_text
from retail_agent.store.definitions import remembered
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


def analyst_middleware(settings: Settings) -> list[AgentMiddleware]:
    """The stack that bounds the SQL loop.

    `run_limit` is computed from the same settings the budgets were computed
    from, rather than written down, so tuning one in config moves the agent with
    it. A loop silently allowed twice the queries would look like better
    accuracy for a reason no report would mention.
    """
    sql_budget = (
        settings.max_analysis_steps + settings.repair_budget + settings.diagnose_budget
    )

    return [
        *_pii(),
        ToolCallLimitMiddleware(tool_name="run_sql", run_limit=sql_budget),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ToolErrorMiddleware(on_error=describe_failure),
    ]


def supervisor_middleware(
    deps: AgentDeps, capture: TurnCapture, *, ask_for_definitions: bool = False
) -> list[AgentMiddleware]:
    """The stack that bounds the turn.

    `ask_for_definitions` arms the pause on an unsettled term. Off by default
    because a pause needs somebody there to answer it: `seams.ask_once` scores a
    paused turn as unanswered, and the eval cases that turn on an undefined term
    are the brief's own examples. Headless callers keep the analyst's early
    return and the disclosure `assumption_note` forces into the answer.
    """
    return [
        _prompt(deps, capture),
        *_pii(),
        _approval_gate(deps, capture, ask_for_definitions=ask_for_definitions),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ToolErrorMiddleware(on_error=describe_failure),
        _recorder(deps, capture),
    ]


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


def _approval_gate(
    deps: AgentDeps, capture: TurnCapture, *, ask_for_definitions: bool = False
) -> AgentMiddleware:
    """The two places a turn stops for a person, as interrupts before the tool.

    Both `when` predicates resolve read-only, without a model call, and park
    what they found on the capture. That is what makes each one a gate rather
    than advice: a delete that matches nothing never raises a prompt, a question
    whose terms are settled is never paused, and in both cases what the user is
    shown is exactly what the predicate found.
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

    def needs_definition(request) -> bool:
        args = request.tool_call.get("args", {})
        # Only ask if the answer can be kept. Without a store the agent would
        # ask the same person the same question every turn, which is worse than
        # assuming and saying so — the rule the analyst's early return follows.
        if deps.definitions is None or args.get("assume_undefined"):
            return False

        capture.pending_definition = open_terms(
            deps, capture, args.get("question", "") or ""
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
    if ask_for_definitions:
        interrupt_on["analyst"] = InterruptOnConfig(
            # `edit` is how "decide for me" resolves: the tool call is rewritten
            # with assume_undefined, which is the path that already exists.
            allowed_decisions=["approve", "edit", "reject"],
            description=describe_definition,
            when=needs_definition,
        )

    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on, description_prefix="")


def open_terms(
    deps: AgentDeps, capture: TurnCapture, question: str
) -> PendingDefinition | None:
    """The terms this question turns on that nothing has settled yet.

    The corpus first, then what this user has already been asked, in that order
    — a trio is a reviewed decision by the people who own the numbers and a
    remembered definition is one person's answer, so neither should produce a
    question that has already been answered.

    Never raises: `recall` swallows a retrieval failure, and `remembered`
    returns what it managed to read. An unreachable store costs a question that
    did not need asking, never the turn.
    """
    found = recall(deps, question)
    unsettled = unresolved(question, found)
    known = remembered(deps.definitions, capture.user_id, unsettled)
    still_open = tuple(term for term in unsettled if term not in known)
    return PendingDefinition(terms=still_open) if still_open else None


def describe_open_terms(terms: Sequence[str]) -> str:
    """What the pause is about, in the words `UNDEFINED_TERMS` already uses."""
    return "\n".join(
        f"{term!r} has no agreed definition — {UNDEFINED_TERMS.get(term, 'it is undefined')}"
        for term in terms
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
