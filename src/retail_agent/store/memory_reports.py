"""In-memory ReportStore.

Ships in `src/` rather than in tests because it is also the honest fallback when
Postgres is unreachable: the agent keeps working for the session, and the CLI
says plainly that reports will not survive it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone

from retail_agent.store.reports import AuditEntry, Report


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryReportStore:
    def __init__(self) -> None:
        self._reports: dict[str, Report] = {}
        self._audit: dict[str, AuditEntry] = {}

    def save(
        self, *, owner_id: str, session_id: str, title: str, body: str
    ) -> Report:
        report = Report(
            id=uuid.uuid4().hex[:12],
            owner_id=owner_id,
            session_id=session_id,
            title=title,
            body=body,
            created_at=_now(),
        )
        self._reports[report.id] = report
        return report

    def get(self, *, owner_id: str, report_id: str) -> Report | None:
        report = self._reports.get(report_id)
        return report if report and report.owner_id == owner_id else None

    def list_reports(self, *, owner_id: str, limit: int = 20) -> list[Report]:
        live = [
            r
            for r in self._reports.values()
            if r.owner_id == owner_id and r.deleted_at is None
        ]
        live.sort(key=lambda r: r.created_at, reverse=True)
        return live[:limit]

    def resolve(
        self,
        *,
        owner_id: str,
        term: str | None = None,
        session_id: str | None = None,
    ) -> list[Report]:
        found = self.list_reports(owner_id=owner_id, limit=1_000)
        if session_id is not None:
            found = [r for r in found if r.session_id == session_id]
        if term:
            needle = term.lower()
            found = [
                r
                for r in found
                if needle in r.title.lower() or needle in r.body.lower()
            ]
        return found

    def soft_delete(
        self,
        *,
        owner_id: str,
        report_ids: Sequence[str],
        action_id: str,
        token: str,
    ) -> int:
        if action_id in self._audit:
            return 0  # mirrors the primary key on report_audit.action_id

        deleted: list[str] = []
        for report_id in report_ids:
            report = self.get(owner_id=owner_id, report_id=report_id)
            if report is None or report.deleted_at is not None:
                continue
            self._reports[report_id] = replace(report, deleted_at=_now())
            deleted.append(report_id)

        if not deleted:
            return 0

        self._audit[action_id] = AuditEntry(
            action_id=action_id,
            owner_id=owner_id,
            report_ids=tuple(deleted),
            token=token,
            performed_at=_now(),
        )
        return len(deleted)

    def undo(self, *, owner_id: str, action_id: str | None = None) -> int:
        entry = (
            self._audit.get(action_id)
            if action_id
            else self.last_action(owner_id=owner_id)
        )
        if entry is None or entry.owner_id != owner_id or entry.undone_at:
            return 0

        restored = 0
        for report_id in entry.report_ids:
            report = self._reports.get(report_id)
            if report is None or report.deleted_at is None:
                continue
            self._reports[report_id] = replace(report, deleted_at=None)
            restored += 1

        self._audit[entry.action_id] = replace(entry, undone_at=_now())
        return restored

    def last_action(self, *, owner_id: str) -> AuditEntry | None:
        mine = [
            e
            for e in self._audit.values()
            if e.owner_id == owner_id and e.undone_at is None
        ]
        if not mine:
            return None
        return max(mine, key=lambda e: e.performed_at)
