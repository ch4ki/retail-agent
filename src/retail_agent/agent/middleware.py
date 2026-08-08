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

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import (
    PERSONA_DEFAULT,
    REFUSAL,
    SAFETY_RULES,
    SUPERVISOR_PROMPT,
)
from retail_agent.agent.reports import render_manifest, resolve_delete
from retail_agent.agent.tools import GuardRejection
from retail_agent.config import Settings
from retail_agent.datasources.base import DataSourceError
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text
from retail_agent.safety.scope import refuse
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

# `fresh_scratch` used to default this and the graph spent it outside
# `repair_budget`. Named here rather than read from Settings because it is not
# one — if it becomes a setting, this should follow it.
DIAGNOSE_BUDGET = 1


def analyst_middleware(settings: Settings) -> list[AgentMiddleware]:
    """The stack that bounds the SQL loop.

    `run_limit` is computed from the same settings the budgets were computed
    from, rather than written down, so tuning one in config moves the agent with
    it. A loop silently allowed twice the queries would look like better
    accuracy for a reason no report would mention.
    """
    sql_budget = settings.max_analysis_steps + settings.repair_budget + DIAGNOSE_BUDGET

    return [
        *_pii(),
        ToolCallLimitMiddleware(tool_name="run_sql", run_limit=sql_budget),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ToolErrorMiddleware(on_error=describe_failure),
    ]


def supervisor_middleware(
    deps: AgentDeps, capture: TurnCapture
) -> list[AgentMiddleware]:
    """The stack that bounds the turn."""
    return [
        _scope_guard(),
        _prompt(deps, capture),
        *_pii(),
        _approval_gate(deps, capture),
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

    This is the repair prompt. Only the two failures the graph also recovered
    from are handled; anything else propagates, so an internal bug surfaces as a
    failed turn rather than as an agent that quietly worked around it.

    `request` is the `ToolCallRequest` and is unused, but the parameter is not
    optional: `OnError` is `Callable[[Exception, ToolCallRequest], ...]`, and a
    one-argument version type-checks, imports, and passes every test that does
    not call it — then raises TypeError against the live provider the first time
    a query is rejected. Which is exactly what happened once already: the repair
    path was dead for a whole eval run and reported as an agent failure.
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


def _approval_gate(deps: AgentDeps, capture: TurnCapture) -> AgentMiddleware:
    """The confirmation flow, as an interrupt before the tool runs.

    `when` resolves the target set read-only and stores it, so two things hold:
    a delete that matches nothing never raises a prompt at all, and the manifest
    the user approves is exactly the set the tool then deletes.
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

    return HumanInTheLoopMiddleware(
        interrupt_on={
            "delete_reports": InterruptOnConfig(
                allowed_decisions=["approve", "reject"],
                description=describe,
                when=has_targets,
            )
        },
        description_prefix="",
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
