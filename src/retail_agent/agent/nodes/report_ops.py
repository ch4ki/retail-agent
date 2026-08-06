"""Operations on the saved report library.

Report bodies are synthesised from conversation history rather than from
`frames`, because frames are per-turn scratch and are already blank by the time
a user says "save that".
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message, render_history
from retail_agent.agent.prompts import (
    REPORT_BODY_PROMPT,
    REPORT_OP_PROMPT,
    SAFETY_RULES,
)
from retail_agent.agent.state import PendingAction, TurnState
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text
from retail_agent.store.personas import active_body

log = logging.getLogger(__name__)


class ReportOp(BaseModel):
    """What the user wants done to their saved report library."""

    kind: Literal["save", "list", "delete"] = Field(
        description="save: write the current analysis to a report. "
        "list: show saved reports. delete: remove saved reports."
    )
    title: str = Field("", description="Short title. Only meaningful for save.")
    term: str = Field(
        "", description="Text the target reports mention. Only for delete."
    )
    session_scoped: bool = Field(
        False,
        description="True only when the user means the reports made in this "
        "conversation. Only for delete.",
    )


LIST_ONLY = ReportOp(kind="list")


def extract_op(state: TurnState, deps: AgentDeps) -> ReportOp:
    """Ask for a structured operation, then discard fields that do not apply.

    Constrained decoding makes the model populate *every* field, including ones
    irrelevant to the chosen operation — a live spike returned
    `session_scoped=True` for "what have I saved?". Those fields decide what a
    delete targets, so they are only trusted on a delete.
    """
    try:
        op = deps.llm.with_structured_output(ReportOp).invoke(
            [
                HumanMessage(
                    content=REPORT_OP_PROMPT.format(
                        history=render_history(state, deps.settings.history_messages),
                        question=last_user_message(state),
                    )
                )
            ]
        )
    except Exception as err:
        # Listing is the only harmless default. Guessing `delete` from a reply
        # we could not read is how you delete someone's work.
        log.warning("report op structured output failed (%s); listing", err)
        return LIST_ONLY

    if op.kind != "delete":
        return ReportOp(kind=op.kind, title=op.title)
    return op


def report_ops_node(state: TurnState, deps: AgentDeps) -> dict:
    """Read-only in every branch.

    Save and list answer here. A delete resolves its target set and stages it,
    then stops — the write lives in `apply_delete`, on the far side of the
    confirmation breakpoint. Resolving here rather than in a second node keeps
    the operation the model chose in one place instead of extracting it twice.
    """
    op = extract_op(state, deps)

    if op.kind == "save":
        return _save(state, deps, op)
    if op.kind == "delete":
        return _stage_delete(state, deps, op)
    return _list(state, deps)


def _save(state: TurnState, deps: AgentDeps, op: ReportOp) -> dict:
    history = render_history(state, deps.settings.history_messages)
    body_reply = deps.llm.invoke(
        [
            HumanMessage(
                content=REPORT_BODY_PROMPT.format(
                    persona=active_body(deps.personas), safety=SAFETY_RULES, history=history
                )
            )
        ]
    )
    body = scan_text(message_text(body_reply)).text

    report = deps.reports.save(
        owner_id=state.get("user_id", ""),
        session_id=state.get("session_id", ""),
        title=op.title or "Untitled report",
        body=body,
    )

    answer = (
        f"Saved as **{report.title}** (`{report.id}`).\n\n{body}\n\n"
        f"_`/reports` lists what you have saved._"
    )
    return _answered(answer)


def _list(state: TurnState, deps: AgentDeps) -> dict:
    saved = deps.reports.list_reports(owner_id=state.get("user_id", ""))
    if not saved:
        return _answered("You have no saved reports yet.")

    lines = [f"- **{r.title}** (`{r.id}`) — {r.created_at:%Y-%m-%d}" for r in saved]
    return _answered("Your saved reports:\n\n" + "\n".join(lines))


def _answered(text: str) -> dict:
    return {"answer": text, "status": "ok", "messages": [AIMessage(content=text)]}


def confirmation_token(count: int) -> str:
    """One report is a low-stakes correction; several is not. The typed token
    makes a bulk delete impossible to trigger by reflex."""
    return "y" if count == 1 else f"DELETE {count}"


def _stage_delete(state: TurnState, deps: AgentDeps, op: ReportOp) -> dict:
    """Resolve the target set and stage it for confirmation. Reads only."""
    owner_id = state.get("user_id", "")

    targets = deps.reports.resolve(
        owner_id=owner_id,
        term=op.term or None,
        session_id=state.get("session_id") if op.session_scoped else None,
    )

    if not targets:
        described = f" mentioning '{op.term}'" if op.term else ""
        return _answered(f"I found no reports{described} to delete.")

    return {
        "pending_action": PendingAction(
            action_id=uuid.uuid4().hex[:16],
            report_ids=tuple(r.id for r in targets),
            titles=tuple(r.title for r in targets),
            token=confirmation_token(len(targets)),
        )
    }


def await_confirmation(state: TurnState) -> dict:
    """A deliberate no-op. The graph breaks *before* this node; the CLI fills
    `confirmation` in as though this node had produced it."""
    return {}


def apply_delete(state: TurnState, deps: AgentDeps) -> dict:
    action = state.get("pending_action")
    if action is None:
        return _answered("There is nothing waiting to be deleted.")

    if state.get("confirmation", "").strip() != action.token:
        return _answered(
            f"Aborted — nothing was deleted. "
            f"{len(action.report_ids)} report(s) still there."
        )

    deleted = deps.reports.soft_delete(
        owner_id=state.get("user_id", ""),
        report_ids=action.report_ids,
        action_id=action.action_id,
        token=action.token,
    )

    if deleted == 0:
        return _answered("Nothing to delete — those reports are already gone.")

    return _answered(f"Deleted {deleted} report(s). Run `/undo` to restore them.")
