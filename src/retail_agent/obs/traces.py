"""Turn traces: what the agent did, kept so it can be explained afterwards.

The record is deliberately flat and already-masked. Everything in it comes from
the turn's `TurnCapture`, and the result frame — the only place row values live
— is not among the fields. A trace cannot become a second disclosure path.

Metrics live behind the same protocol as storage because "how often does SQL
pass the guard first time" is a question about stored turns, and answering it in
SQL beats pulling every trace into Python.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# (step, duration_ms, detail) — one tool call. Built by `TurnCapture`, which is
# where a turn's timings are accumulated; nothing here reads agent internals.
Event = tuple[str, int, str]

MAX_ANSWER_CHARS = 4_000


@dataclass(frozen=True)
class TraceRecord:
    turn_id: str
    session_id: str
    owner_id: str
    question: str
    intent: str
    status: str
    answer: str = ""
    redactions: int = 0
    bytes_billed: int = 0
    duration_ms: int = 0
    events: list[Event] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)


@runtime_checkable
class TraceStore(Protocol):
    def record(self, trace: TraceRecord) -> None: ...

    def get(self, *, owner_id: str, turn_id: str) -> TraceRecord | None: ...

    def recent(self, *, owner_id: str, limit: int = 20) -> list[TraceRecord]: ...

    def metrics(self, *, owner_id: str, limit: int = 200) -> dict: ...


def compute_metrics(traces: list[TraceRecord]) -> dict:
    """The agent-level metrics, derived from stored turns.

    Shared by both stores so the numbers cannot differ by implementation.
    """
    if not traces:
        return {
            "turns": 0,
            "degraded_rate": 0.0,
            "first_pass_validity": 0.0,
            "self_correction_rate": 0.0,
            "redactions": 0,
            "bytes_billed": 0,
            "node_p50_ms": {},
        }

    with_sql = [t for t in traces if t.attempts]
    first_clean = [
        t
        for t in with_sql
        if not t.attempts[0].get("violations") and not t.attempts[0].get("error")
    ]
    needed_repair = [t for t in with_sql if len(t.attempts) > 1]
    recovered = [t for t in needed_repair if t.status == "ok"]

    by_node: dict[str, list[int]] = {}
    for trace in traces:
        for node, duration, _ in trace.events:
            by_node.setdefault(node, []).append(duration)

    return {
        "turns": len(traces),
        "degraded_rate": _ratio(
            len([t for t in traces if t.status == "degraded"]), len(traces)
        ),
        "first_pass_validity": _ratio(len(first_clean), len(with_sql)),
        "self_correction_rate": _ratio(len(recovered), len(needed_repair)),
        "redactions": sum(t.redactions for t in traces),
        "bytes_billed": sum(t.bytes_billed for t in traces),
        "node_p50_ms": {
            node: int(statistics.median(times)) for node, times in sorted(by_node.items())
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


class InMemoryTraceStore:
    """Used when Postgres is unreachable, and by the component tests.

    `/trace` still works for the running session; only history across restarts
    is lost.
    """

    def __init__(self) -> None:
        self._traces: dict[str, TraceRecord] = {}
        self._order: list[str] = []

    def record(self, trace: TraceRecord) -> None:
        if trace.turn_id not in self._traces:
            self._order.append(trace.turn_id)
        self._traces[trace.turn_id] = trace

    def get(self, *, owner_id: str, turn_id: str) -> TraceRecord | None:
        trace = self._traces.get(turn_id)
        return trace if trace and trace.owner_id == owner_id else None

    def recent(self, *, owner_id: str, limit: int = 20) -> list[TraceRecord]:
        mine = [
            self._traces[turn_id]
            for turn_id in reversed(self._order)
            if self._traces[turn_id].owner_id == owner_id
        ]
        return mine[:limit]

    def metrics(self, *, owner_id: str, limit: int = 200) -> dict:
        return compute_metrics(self.recent(owner_id=owner_id, limit=limit))


class PostgresTraceStore:
    """Traces in `traces`, node timings in `turn_events`.

    A turn is written once, as one transaction: the summary upserted by
    `turn_id` and its events replaced. Re-recording a turn therefore corrects it
    rather than duplicating it, which matters because every per-turn metric
    divides by the row count.
    """

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def record(self, trace: TraceRecord) -> None:
        from sqlalchemy import delete
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import TraceRow, TurnEventRow

        values = {
            "turn_id": trace.turn_id,
            "session_id": trace.session_id,
            "owner_id": trace.owner_id,
            "question": trace.question,
            "intent": trace.intent,
            "status": trace.status,
            "answer": trace.answer,
            "redactions": trace.redactions,
            "bytes_billed": trace.bytes_billed,
            "duration_ms": trace.duration_ms,
            "attempts": trace.attempts,
        }
        with self._sessions.begin() as session:
            session.execute(
                pg_insert(TraceRow)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["turn_id"],
                    set_={k: v for k, v in values.items() if k != "turn_id"},
                )
            )
            session.execute(
                delete(TurnEventRow).where(TurnEventRow.turn_id == trace.turn_id)
            )
            session.add_all(
                [
                    TurnEventRow(
                        turn_id=trace.turn_id,
                        seq=seq,
                        node=node,
                        duration_ms=duration,
                        detail=detail,
                    )
                    for seq, (node, duration, detail) in enumerate(trace.events)
                ]
            )

    def get(self, *, owner_id: str, turn_id: str) -> TraceRecord | None:
        from sqlalchemy import select

        from retail_agent.store.models import TraceRow

        with self._sessions() as session:
            row = session.scalar(
                select(TraceRow).where(
                    TraceRow.turn_id == turn_id, TraceRow.owner_id == owner_id
                )
            )
            if row is None:
                return None
            return self._to_record(session, row)

    def recent(self, *, owner_id: str, limit: int = 20) -> list[TraceRecord]:
        from sqlalchemy import select

        from retail_agent.store.models import TraceRow

        with self._sessions() as session:
            rows = session.scalars(
                select(TraceRow)
                .where(TraceRow.owner_id == owner_id)
                .order_by(TraceRow.created_at.desc(), TraceRow.turn_id.desc())
                .limit(limit)
            ).all()
            return [self._to_record(session, row) for row in rows]

    def metrics(self, *, owner_id: str, limit: int = 200) -> dict:
        return compute_metrics(self.recent(owner_id=owner_id, limit=limit))

    @staticmethod
    def _to_record(session, row) -> TraceRecord:
        from sqlalchemy import select

        from retail_agent.store.models import TurnEventRow

        events = session.scalars(
            select(TurnEventRow)
            .where(TurnEventRow.turn_id == row.turn_id)
            .order_by(TurnEventRow.seq)
        ).all()
        return TraceRecord(
            turn_id=row.turn_id,
            session_id=row.session_id,
            owner_id=row.owner_id,
            question=row.question,
            intent=row.intent,
            status=row.status,
            answer=row.answer,
            redactions=row.redactions,
            bytes_billed=row.bytes_billed,
            duration_ms=row.duration_ms,
            events=[(e.node, e.duration_ms, e.detail) for e in events],
            attempts=list(row.attempts or []),
        )


def build_trace_store(settings, on_degraded=None) -> TraceStore:
    """Postgres when it is reachable, memory when it is not.

    Same bargain as the report store and the checkpointer: losing the database
    costs history across restarts, never the ability to use the agent.
    """
    from retail_agent.store.db import sessions_or_none

    sessions = sessions_or_none(
        settings.database_url, name="trace store", on_degraded=on_degraded
    )
    return PostgresTraceStore(sessions) if sessions else InMemoryTraceStore()
