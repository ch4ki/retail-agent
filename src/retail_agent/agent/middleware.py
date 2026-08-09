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
    before_agent,
    dynamic_prompt,
)
from langchain_core.messages import AIMessage

from retail_agent.agent.capture import PendingDefinition, TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import (
    PERSONA_DEFAULT,
    REFUSAL,
    SAFETY_RULES,
    SUPERVISOR_PROMPT,
)
from retail_agent.agent.reports import render_manifest, resolve_delete
from retail_agent.agent.tools import GuardRejection, recall
from retail_agent.config import Settings
from retail_agent.datasources.base import DataSourceError
from retail_agent.knowledge.trios import UNDEFINED_TERMS, unresolved
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text
from retail_agent.safety.scope import refuse
from retail_agent.store.definitions import remembered
from retail_agent.store.personas import active_body
from retail_agent.store.preferences import preferred, style_instruction

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
        _scope_guard(),
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


def _prompt(deps: AgentDeps, capture: TurnCapture) -> AgentMiddleware:
    """The supervisor's system prompt, assembled per model call.

    Per call rather than at build time, and that is not a detail. The persona is
    a database row the CEO can edit weekly, and the preference is a row the user
    can change mid-session; binding either when the agent is constructed means a
    long-lived process serves the old one until it restarts.
    """

    @dynamic_prompt
    def supervisor_prompt(request) -> str:
        return SUPERVISOR_PROMPT.format(
            persona=active_body(deps.personas) or PERSONA_DEFAULT,
            safety=SAFETY_RULES,
        ) + "\n\n" + style_instruction(preferred(deps.preferences, capture.user_id))

    return supervisor_prompt


def _scope_guard() -> AgentMiddleware:
    """Refuse before a model reads it.

    `can_jump_to=["end"]` is what makes this a guard rather than advice: the
    refusal is the whole turn, and no tool can run after it.
    """

    @before_agent(can_jump_to=["end"])
    def scope_guard(state, runtime) -> dict | None:
        question = _last_user_text(state)
        reason = refuse(question) if question else None
        if reason is None:
            return None

        log.info("refused a request: %s", reason)
        return {
            "messages": [AIMessage(content=f"{REFUSAL}\n\n(Specifically: {reason}.)")],
            "jump_to": "end",
        }

    return scope_guard


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
    """The last sweep and the trace, in one hook.

    Both, rather than two hooks, because their order matters and `after_agent`
    hooks run in the reverse of the list they were declared in — an ordering
    that is easy to get right once and easy to break silently later. The stored
    trace must carry the text the user saw, so the scan happens first, here.
    """

    @after_agent
    def record(state, runtime) -> dict | None:
        message = _last_ai_message(state)
        answer = message_text(message).strip() if message is not None else ""

        scanned = scan_text(answer)
        if scanned.findings:
            log.warning("egress scan redacted %s", ", ".join(scanned.findings))
            capture.status = "degraded"

        try:
            deps.traces.record(capture.to_trace(scanned.text))
        except Exception as err:
            # Losing a trace should not lose the answer it describes.
            log.warning("could not record the trace (%s)", err)

        if not scanned.findings or message is None:
            return None
        # Same id, so the reducer replaces the message rather than appending a
        # second copy of the answer.
        return {"messages": [AIMessage(content=scanned.text, id=message.id)]}

    return record


def _last_user_text(state: dict) -> str:
    for message in reversed(state.get("messages", []) or []):
        if getattr(message, "type", "") == "human":
            return message_text(message).strip()
    return ""


def _last_ai_message(state: dict):
    for message in reversed(state.get("messages", []) or []):
        if isinstance(message, AIMessage) and message_text(message).strip():
            return message
    return None
