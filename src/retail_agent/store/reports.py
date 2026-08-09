"""Saved reports: the domain types, the storage protocol, and both stores.

Ownership and replay safety are properties of the SQL, not of the calling code.
Every query filters on owner_id; `report_audit.action_id` is a primary key, so a
replayed confirmation inserts nothing and therefore deletes nothing. The
in-memory store mirrors those guarantees in Python so the two are
interchangeable behind `ReportStore`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from sqlalchemy import delete, func, literal, select, update
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from retail_agent.store.models import ReportAuditRow, ReportRow


# Long enough for a container still starting, short enough that a dead
# database does not stall the CLI.


@dataclass(frozen=True)
class Report:
    id: str
    owner_id: str
    session_id: str
    title: str
    body: str
    created_at: datetime
    deleted_at: datetime | None = None


@dataclass(frozen=True)
class AuditEntry:
    action_id: str
    owner_id: str
    report_ids: tuple[str, ...]
    token: str
    performed_at: datetime
    undone_at: datetime | None = None


@runtime_checkable
class ReportStore(Protocol):
    def save(
        self, *, owner_id: str, session_id: str, title: str, body: str
    ) -> Report: ...

    def get(self, *, owner_id: str, report_id: str) -> Report | None: ...

    def list_reports(self, *, owner_id: str, limit: int = 20) -> list[Report]: ...

    def resolve(
        self,
        *,
        owner_id: str,
        term: str | None = None,
        session_id: str | None = None,
    ) -> list[Report]: ...

    def soft_delete(
        self,
        *,
        owner_id: str,
        report_ids: Sequence[str],
        action_id: str,
        token: str,
    ) -> int: ...

    def undo(self, *, owner_id: str, action_id: str | None = None) -> int: ...

    def last_action(self, *, owner_id: str) -> AuditEntry | None: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryReportStore:
    """The honest fallback when Postgres is unreachable.

    Not a test double: the agent keeps working for the session, and the CLI says
    plainly that reports will not survive it.
    """

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


def _to_report(row) -> Report:
    """Persistence row -> domain type. Bound by name, so adding a column or
    reordering a query cannot silently shift values between fields."""
    return Report(
        id=row.id,
        owner_id=row.owner_id,
        session_id=row.session_id,
        title=row.title,
        body=row.body,
        created_at=row.created_at,
        deleted_at=row.deleted_at,
    )


def _to_audit(row) -> AuditEntry:
    return AuditEntry(
        action_id=row.action_id,
        owner_id=row.owner_id,
        report_ids=tuple(row.report_ids),
        token=row.token,
        performed_at=row.performed_at,
        undone_at=row.undone_at,
    )


class PostgresReportStore:
    """Every statement carries `owner_id`. That predicate is the whole
    access-control model, and it holds even if the model is compromised.
    """

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def save(
        self, *, owner_id: str, session_id: str, title: str, body: str
    ) -> Report:
        row = ReportRow(
            id=uuid.uuid4().hex[:12],
            owner_id=owner_id,
            session_id=session_id,
            title=title,
            body=body,
        )
        with self._sessions.begin() as session:
            session.add(row)
            session.flush()
            session.refresh(row)  # pick up server-side created_at
        return _to_report(row)

    def get(self, *, owner_id: str, report_id: str) -> Report | None:
        with self._sessions() as session:
            row = session.scalar(
                select(ReportRow).where(
                    ReportRow.id == report_id, ReportRow.owner_id == owner_id
                )
            )
        return _to_report(row) if row else None

    def list_reports(self, *, owner_id: str, limit: int = 20) -> list[Report]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ReportRow)
                .where(ReportRow.owner_id == owner_id, ReportRow.deleted_at.is_(None))
                .order_by(ReportRow.created_at.desc())
                .limit(limit)
            ).all()
        return [_to_report(row) for row in rows]

    def resolve(
        self,
        *,
        owner_id: str,
        term: str | None = None,
        session_id: str | None = None,
    ) -> list[Report]:
        query = select(ReportRow).where(
            ReportRow.owner_id == owner_id, ReportRow.deleted_at.is_(None)
        )
        if session_id is not None:
            query = query.where(ReportRow.session_id == session_id)
        if term:
            # Full-text search in the database. The model supplies the term; it
            # never decides which reports match.
            query = query.where(
                func.to_tsvector(
                    literal("english", type_=REGCONFIG),
                    ReportRow.title + literal(" ") + ReportRow.body,
                ).bool_op("@@")(
                    func.plainto_tsquery(literal("english", type_=REGCONFIG), term)
                )
            )

        with self._sessions() as session:
            rows = session.scalars(query.order_by(ReportRow.created_at.desc())).all()
        return [_to_report(row) for row in rows]

    def soft_delete(
        self,
        *,
        owner_id: str,
        report_ids: Sequence[str],
        action_id: str,
        token: str,
    ) -> int:
        with self._sessions.begin() as session:
            # Claiming the action id first is what makes a replayed resume a
            # no-op: the conflict inserts nothing, so nothing is deleted.
            claimed = session.execute(
                pg_insert(ReportAuditRow)
                .values(
                    action_id=action_id,
                    owner_id=owner_id,
                    report_ids=list(report_ids),
                    token=token,
                )
                .on_conflict_do_nothing(index_elements=["action_id"])
                .returning(ReportAuditRow.action_id)
            ).scalar()

            if claimed is None:
                return 0

            deleted = session.scalars(
                update(ReportRow)
                .where(
                    ReportRow.id.in_(list(report_ids)),
                    ReportRow.owner_id == owner_id,
                    ReportRow.deleted_at.is_(None),
                )
                .values(deleted_at=func.now())
                .returning(ReportRow.id)
            ).all()

            if not deleted:
                # Nothing was ours to delete. Drop the claim so the audit trail
                # records only actions that actually happened.
                session.execute(
                    delete(ReportAuditRow).where(
                        ReportAuditRow.action_id == action_id
                    )
                )
                return 0

            session.execute(
                update(ReportAuditRow)
                .where(ReportAuditRow.action_id == action_id)
                .values(report_ids=list(deleted))
            )
        return len(deleted)

    def undo(self, *, owner_id: str, action_id: str | None = None) -> int:
        entry = (
            self._entry(action_id=action_id, owner_id=owner_id)
            if action_id
            else self.last_action(owner_id=owner_id)
        )
        if entry is None:
            return 0

        with self._sessions.begin() as session:
            restored = session.scalars(
                update(ReportRow)
                .where(
                    ReportRow.id.in_(list(entry.report_ids)),
                    ReportRow.owner_id == owner_id,
                    ReportRow.deleted_at.is_not(None),
                )
                .values(deleted_at=None)
                .returning(ReportRow.id)
            ).all()
            session.execute(
                update(ReportAuditRow)
                .where(ReportAuditRow.action_id == entry.action_id)
                .values(undone_at=func.now())
            )
        return len(restored)

    def last_action(self, *, owner_id: str) -> AuditEntry | None:
        with self._sessions() as session:
            row = session.scalar(
                select(ReportAuditRow)
                .where(
                    ReportAuditRow.owner_id == owner_id,
                    ReportAuditRow.undone_at.is_(None),
                )
                .order_by(ReportAuditRow.performed_at.desc())
                .limit(1)
            )
        return _to_audit(row) if row else None

    def _entry(self, *, action_id: str, owner_id: str) -> AuditEntry | None:
        with self._sessions() as session:
            row = session.scalar(
                select(ReportAuditRow).where(
                    ReportAuditRow.action_id == action_id,
                    ReportAuditRow.owner_id == owner_id,
                    ReportAuditRow.undone_at.is_(None),
                )
            )
        return _to_audit(row) if row else None


def build_report_store(settings, on_degraded=None) -> ReportStore:
    """Postgres when it is reachable, memory when it is not.

    Mirrors how the checkpointer degrades: an unreachable database costs
    durability, not the ability to use the agent. The probe also catches the
    case where Postgres is up but migrations have never been run.
    """
    from retail_agent.store.db import sessions_or_none

    sessions = sessions_or_none(
        settings.database_url,
        name="report store",
        on_degraded=on_degraded,
        # A stronger probe than a bare connect: this also catches Postgres being
        # up but never migrated, which otherwise fails on the first save.
        probe=lambda conn: conn.execute(select(ReportRow.id).limit(1)),
    )
    return PostgresReportStore(sessions) if sessions else InMemoryReportStore()
