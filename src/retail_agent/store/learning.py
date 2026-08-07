"""Learning a user's preferences from how they ask.

The brief asks the agent to learn over time — how deep someone likes the
analysis, whether they want tables or prose. Two rules shape this:

**Proposed, never applied silently.** Personalisation that changes the output
without saying why is worse than none, because the reader cannot tell whether
the agent changed or the data did. Evidence accumulates; the agent asks.

**The evidence is quotable.** The proposal says *you asked for this three times,
most recently "cut to the chase"* — a span the user actually typed. "The model
inferred you prefer brevity" is not something anyone can check or argue with.

Detection itself lives in the router (`agent/nodes/route.py`), folded into the
model call that already reads the question. It used to be regex here, and the
regex was wrong in a way that mattered: it caught about a quarter of realistic
phrasings, and it fired backwards on negation — "don't just give me the number,
tell me why" recorded `depth=summary`, then quoted the user out of context. What
survived the move is the guarantee that the quote is real: the router discards
any signal whose evidence is not verbatim in the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Evidence needed before the agent asks. Two is a coincidence.
PROPOSAL_THRESHOLD = 3

# After a decline, the same suggestion needs a lot more evidence — being asked
# twice about something you already refused is how a helpful feature becomes an
# irritating one.
DECLINED_MULTIPLIER = 3


@dataclass(frozen=True)
class Signal:
    field: str
    value: str
    evidence: str  # the phrase the user actually used


@dataclass(frozen=True)
class Proposal:
    field: str
    value: str
    count: int
    evidence: str

    def question(self) -> str:
        return (
            f"You've asked for this {self.count} times — most recently "
            f'"{self.evidence}". Set {self.field} to {self.value}? '
            f"[bold]/prefs accept[/bold] or [bold]/prefs decline[/bold]"
        )


@runtime_checkable
class SignalStore(Protocol):
    def record(self, *, user_id: str, signal: Signal) -> int: ...

    def counts(self, *, user_id: str) -> dict[tuple[str, str], tuple[int, str]]: ...

    def decline(self, *, user_id: str, field: str, value: str) -> None: ...

    def declines(self, *, user_id: str) -> dict[tuple[str, str], int]: ...

    def clear(self, *, user_id: str, field: str) -> None: ...


class InMemorySignalStore:
    def __init__(self) -> None:
        self._counts: dict[str, dict[tuple[str, str], tuple[int, str]]] = {}
        self._declines: dict[str, dict[tuple[str, str], int]] = {}

    def record(self, *, user_id: str, signal: Signal) -> int:
        mine = self._counts.setdefault(user_id, {})
        key = (signal.field, signal.value)
        count = mine.get(key, (0, ""))[0] + 1
        mine[key] = (count, signal.evidence)
        return count

    def counts(self, *, user_id: str) -> dict[tuple[str, str], tuple[int, str]]:
        return dict(self._counts.get(user_id, {}))

    def decline(self, *, user_id: str, field: str, value: str) -> None:
        mine = self._declines.setdefault(user_id, {})
        mine[(field, value)] = mine.get((field, value), 0) + 1

    def declines(self, *, user_id: str) -> dict[tuple[str, str], int]:
        return dict(self._declines.get(user_id, {}))

    def clear(self, *, user_id: str, field: str) -> None:
        """Once a field is settled, its evidence stops mattering — otherwise
        accepting a proposal leaves the counters that produced it in place."""
        mine = self._counts.get(user_id, {})
        for key in [k for k in mine if k[0] == field]:
            del mine[key]


class PostgresSignalStore:
    """Evidence that survives the process.

    Upserts rather than read-modify-write: two terminals for the same user are
    ordinary, and losing a sighting to a lost update would silently move the
    threshold.
    """

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def record(self, *, user_id: str, signal: Signal) -> int:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import PreferenceSignalRow

        statement = (
            pg_insert(PreferenceSignalRow)
            .values(
                user_id=user_id,
                field=signal.field,
                value=signal.value,
                count=1,
                evidence=signal.evidence,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "field", "value"],
                set_={
                    "count": PreferenceSignalRow.count + 1,
                    # The newest wording wins: the proposal says "most
                    # recently", and quoting something from three sessions ago
                    # reads as stale.
                    "evidence": signal.evidence,
                },
            )
            .returning(PreferenceSignalRow.count)
        )
        with self._sessions.begin() as session:
            return int(session.execute(statement).scalar_one())

    def counts(self, *, user_id: str) -> dict[tuple[str, str], tuple[int, str]]:
        from sqlalchemy import select

        from retail_agent.store.models import PreferenceSignalRow

        with self._sessions.begin() as session:
            rows = session.execute(
                select(
                    PreferenceSignalRow.field,
                    PreferenceSignalRow.value,
                    PreferenceSignalRow.count,
                    PreferenceSignalRow.evidence,
                ).where(PreferenceSignalRow.user_id == user_id)
            ).all()
        return {(f, v): (count, evidence) for f, v, count, evidence in rows}

    def decline(self, *, user_id: str, field: str, value: str) -> None:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import PreferenceDeclineRow

        with self._sessions.begin() as session:
            session.execute(
                pg_insert(PreferenceDeclineRow)
                .values(user_id=user_id, field=field, value=value, count=1)
                .on_conflict_do_update(
                    index_elements=["user_id", "field", "value"],
                    set_={"count": PreferenceDeclineRow.count + 1},
                )
            )

    def declines(self, *, user_id: str) -> dict[tuple[str, str], int]:
        from sqlalchemy import select

        from retail_agent.store.models import PreferenceDeclineRow

        with self._sessions.begin() as session:
            rows = session.execute(
                select(
                    PreferenceDeclineRow.field,
                    PreferenceDeclineRow.value,
                    PreferenceDeclineRow.count,
                ).where(PreferenceDeclineRow.user_id == user_id)
            ).all()
        return {(f, v): count for f, v, count in rows}

    def clear(self, *, user_id: str, field: str) -> None:
        """Evidence only. A decline outlives the setting it argued against —
        forgetting it would let the next proposal arrive at full strength."""
        from sqlalchemy import delete

        from retail_agent.store.models import PreferenceSignalRow

        with self._sessions.begin() as session:
            session.execute(
                delete(PreferenceSignalRow).where(
                    PreferenceSignalRow.user_id == user_id,
                    PreferenceSignalRow.field == field,
                )
            )


def build_signal_store(settings, on_degraded=None):
    """Postgres when reachable, memory when not.

    Degrading costs the learning loop, not the agent: without a database the
    counts simply never reach the threshold, which is what happened everywhere
    before this existed.
    """
    import logging

    from retail_agent.store.db import create_db_engine, session_factory

    try:
        engine = create_db_engine(settings.database_url)
        with engine.connect():
            pass
        return PostgresSignalStore(session_factory(engine))
    except Exception as err:
        logging.getLogger(__name__).debug("signal store degraded: %s", err)
        if on_degraded is not None:
            on_degraded()
        return InMemorySignalStore()


def next_proposal(
    signals: SignalStore,
    *,
    user_id: str,
    current,
    threshold: int = PROPOSAL_THRESHOLD,
) -> Proposal | None:
    """The strongest suggestion worth making, or nothing.

    Nothing is proposed for a setting the user already has, and a suggestion
    they declined needs several times the evidence before it comes back.
    """
    declines = signals.declines(user_id=user_id)
    best: Proposal | None = None

    for (field, value), (count, evidence) in signals.counts(user_id=user_id).items():
        if getattr(current, field, None) == value:
            continue
        needed = threshold * (DECLINED_MULTIPLIER ** declines.get((field, value), 0))
        if count < needed:
            continue
        if best is None or count > best.count:
            best = Proposal(field=field, value=value, count=count, evidence=evidence)

    return best
