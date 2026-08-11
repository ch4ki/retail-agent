"""The saved report library, as tools.

`delete_reports` is the destructive one and the only tool behind the human
approval gate. The gate is `HumanInTheLoopMiddleware`, which interrupts *before*
the tool runs — so the write cannot have happened when the user is asked. The
graph got the same property by putting `apply_delete` on the far side of a
breakpoint; this gets it with one fewer node and no state to carry across it.

Listing is not gated. The brief asks for a strict confirmation flow "without
breaking UX", and confirming a read is how a confirmation prompt becomes
something people click through without looking. Saving is no longer a tool at
all — `report_writer` stores what it wrote, so there is no second copy of the
text for a model to alter on its way to the library.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.tools import BaseTool, tool

from retail_agent.agent.capture import PendingDelete, TurnCapture
from retail_agent.agent.deps import AgentDeps

log = logging.getLogger(__name__)


def confirmation_token(count: int) -> str:
    """One report is a low-stakes correction; several is not. The typed token
    makes a bulk delete impossible to trigger by reflex."""
    return "y" if count == 1 else f"DELETE {count}"


def resolve_delete(
    deps: AgentDeps, capture: TurnCapture, *, term: str, session_scoped: bool
) -> PendingDelete | None:
    """What a delete would take, resolved read-only.

    Called from the interrupt's `when` predicate so the manifest the user
    approves is the manifest that gets deleted, and so a delete matching
    nothing never raises an approval prompt at all.
    """
    targets = deps.reports.resolve(
        owner_id=capture.user_id,
        term=term or None,
        session_id=capture.session_id if session_scoped else None,
    )
    if not targets:
        return None

    return PendingDelete(
        action_id=uuid.uuid4().hex[:16],
        report_ids=tuple(report.id for report in targets),
        titles=tuple(report.title for report in targets),
        token=confirmation_token(len(targets)),
    )


def render_manifest(pending: PendingDelete) -> str:
    """What the user is shown before approving.

    Titles rather than ids: "delete all reports mentioning Client X" is only
    safe to approve if you can see which reports that turned out to mean.
    """
    listed = "\n".join(f"  - {title}" for title in pending.titles)
    return (
        f"This will permanently delete {len(pending.report_ids)} report(s):\n"
        f"{listed}\n\nType {pending.token!r} to confirm, or anything else to cancel."
    )


def build_report_tools(deps: AgentDeps, capture: TurnCapture) -> list[BaseTool]:
    """The library tools, bound to one turn's owner and session."""

    @tool
    def list_reports() -> str:
        """List the reports this executive has saved."""
        with capture.step("list_reports") as step:
            saved = deps.reports.list_reports(owner_id=capture.user_id)
            step.detail = f"{len(saved)} report(s)"
            if not saved:
                return "You have no saved reports yet."
            return "Saved reports:\n" + "\n".join(
                f"- {r.title} (id {r.id}, saved {r.created_at:%Y-%m-%d})"
                for r in saved
            )

    @tool
    def delete_reports(term: str = "", session_scoped: bool = False) -> str:
        """Delete saved reports. Destructive, and confirmed with the user first.

        `term` selects reports whose text mentions it — leave it empty only if
        the executive means all of their reports. Set `session_scoped` only when
        they mean the reports made in this conversation.
        """
        with capture.step("delete_reports") as step:
            # Resolved by the approval gate, so what was shown is what goes. A
            # resolution here would reopen the gap between manifest and act.
            pending = capture.pending or resolve_delete(
                deps, capture, term=term, session_scoped=session_scoped
            )
            if pending is None:
                described = f" mentioning '{term}'" if term else ""
                step.detail = "nothing matched"
                return f"I found no reports{described} to delete."

            deleted = deps.reports.soft_delete(
                owner_id=capture.user_id,
                report_ids=pending.report_ids,
                action_id=pending.action_id,
                token=pending.token,
            )
            capture.pending = None
            step.detail = f"deleted {deleted}"
            if deleted == 0:
                return "Nothing to delete — those reports are already gone."
            return (
                f"Deleted {deleted} report(s). Tell the executive they can run "
                f"/undo to restore them."
            )

    return [list_reports, delete_reports]
