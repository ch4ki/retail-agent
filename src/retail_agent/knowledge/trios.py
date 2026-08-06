"""The Golden Bucket: how analysts here decided what a question means.

The brief's example questions cannot be answered from the schema alone. "Why
are users in state X underspending?" — underspending relative to what? "Why did
churn spike?" — theLook has no subscriptions, no cancellations; churn is not
merely undefined, it cannot be read off the columns at all. Someone decided it
means "ordered before, nothing in the trailing 90 days".

Without that, the agent still answers. It picks a definition, writes clean SQL,
and returns a confident number that does not match what Finance reports, with
nothing in the output showing a guess was made. That silent failure is what
this package exists to prevent.

A trio holds more than three strings. `metric_definitions` is the field that
carries the value: injecting a past *query* copies old date filters and joins
into a question they do not fit, while injecting the *definition* carries the
analyst's judgement and lets the agent write fresh SQL.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, runtime_checkable

# Words that look like columns and are not. Each one is a business judgement
# somebody has to have made; the agent must never quietly make it alone.
#
# Deliberately conservative. A term here that is actually unambiguous costs an
# unnecessary caveat; a term missing from here costs a confident wrong number.
UNDEFINED_TERMS: dict[str, str] = {
    "churn": "which customers count as churned, and over what window",
    "churned": "which customers count as churned, and over what window",
    "underspending": "what spend level counts as low, and compared to whom",
    "overspending": "what spend level counts as high, and compared to whom",
    "top": "how many, and ranked by which measure",
    "best": "ranked by which measure",
    "worst": "ranked by which measure",
    "healthy": "what threshold makes a figure healthy",
    "unhealthy": "what threshold makes a figure unhealthy",
    "performing well": "what counts as performing well",
    "underperforming": "what counts as underperforming",
    "at risk": "what puts a customer at risk, and over what window",
    "loyal": "what makes a customer loyal",
    "engaged": "what makes a customer engaged",
    "active": "what recency makes a customer active",
    "inactive": "what recency makes a customer inactive",
    "high value": "what makes a customer high value",
    "recently": "how many days counts as recent",
    "lately": "how many days counts as recent",
}

_TERM_PATTERNS = tuple(
    (term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
    for term in UNDEFINED_TERMS
)


@dataclass(frozen=True)
class Trio:
    """Question → SQL → report, plus the judgement that connects them."""

    id: str
    question: str
    sql: str
    report: str
    # The part that actually gets reused. {"churn": "ordered in the prior 180
    # days and nothing in the trailing 90, excluding cancelled and returned"}
    metric_definitions: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    author: str = "analyst"
    approved_at: datetime | None = None
    version: int = 1
    # Definitions change. Superseding rather than editing means a report from
    # last quarter can still be read against the definition that produced it.
    superseded_by: str | None = None

    def defines(self, term: str) -> bool:
        return term.lower() in {k.lower() for k in self.metric_definitions}


def undefined_terms(question: str) -> list[str]:
    """Business terms in the question that the schema cannot settle.

    Longer phrases win over the words inside them, so "performing well" is one
    finding rather than also matching nothing else. Order follows the question,
    so the caveat reads in the order the user wrote.
    """
    found: list[tuple[int, str]] = []
    for term, pattern in _TERM_PATTERNS:
        match = pattern.search(question)
        if match:
            found.append((match.start(), term))

    found.sort()
    kept: list[str] = []
    for _, term in found:
        # "top" inside "performing well" would be nonsense, but "at risk"
        # contains "risk" and "high value" contains "value" — drop any term
        # that is a substring of one already kept.
        if any(term in other and term != other for other in kept):
            continue
        kept = [k for k in kept if not (k in term and k != term)]
        kept.append(term)
    return kept


def unresolved(question: str, trios: list[Trio]) -> list[str]:
    """Terms the question raises that no retrieved trio defines.

    This is what the graph branches on. A term with a definition behind it is
    answered from the analyst's judgement; a term without one has to be
    surfaced, because the alternative is inventing it silently.
    """
    return [
        term
        for term in undefined_terms(question)
        if not any(trio.defines(term) for trio in trios)
    ]


def definitions_block(trios: list[Trio]) -> str:
    """The definitions to put in front of the model, deduplicated.

    Only definitions — not the past SQL. A previous query carries its own date
    filters and joins, and pasting it into a new question is how an agent
    answers last quarter's question with this quarter's label.
    """
    seen: dict[str, str] = {}
    for trio in trios:
        for term, meaning in trio.metric_definitions.items():
            seen.setdefault(term.lower(), meaning)

    if not seen:
        return ""

    lines = "\n".join(f"- {term}: {meaning}" for term, meaning in sorted(seen.items()))
    return (
        "Definitions agreed by the analytics team. Use these exactly; do not "
        f"substitute your own:\n{lines}"
    )


def style_examples(trios: list[Trio], limit: int = 2) -> str:
    """How analysts here actually write a finding.

    The `report` field is hard to specify and easy to demonstrate: split by
    cohort, compare against a baseline, close with numbered actions.
    """
    reports = [trio.report.strip() for trio in trios if trio.report.strip()][:limit]
    if not reports:
        return ""
    joined = "\n\n---\n\n".join(reports)
    return f"How analysts here write findings. Match this shape, not this content:\n\n{joined}"


def assumption_note(terms: list[str]) -> str:
    """What the agent says when it had to decide something itself.

    Stated in the answer rather than logged, because the number is only
    trustworthy if the reader knows which judgement produced it.
    """
    if not terms:
        return ""
    described = "; ".join(f"**{term}** — {UNDEFINED_TERMS[term]}" for term in terms)
    return (
        "Before the answer, state in one sentence that no agreed definition "
        f"exists for: {described}. Then give the concrete rule your query "
        "actually applied — a threshold, a window or a ranking the reader "
        'could check, such as "customers with 3 or more orders". Naming the '
        "column or the alias is not a definition and does not help anyone. "
        "Do not apologise, and do not refuse to answer."
    )


def sql_assumption_note(terms: list[str]) -> str:
    """What the SQL writer needs when a term has no agreed definition.

    Different advice from `assumption_note`, which tells the narrator to
    disclose the choice. Here the model has to *make* one: given no threshold,
    it will otherwise reach for a bind parameter — `HAVING COUNT(*) >
    @threshold` — which nothing binds and BigQuery rejects. That happened in a
    live session and cost the entire repair budget to the same 400.
    """
    if not terms:
        return ""
    described = "; ".join(f"{term} ({UNDEFINED_TERMS[term]})" for term in terms)
    return (
        f"No agreed definition exists for: {described}. Choose one concrete, "
        "defensible value and write it as a LITERAL in the query — never a "
        "query parameter such as @threshold, ? or :name, because nothing binds "
        "them and the query will fail. Prefer a round, explainable number."
    )


@runtime_checkable
class TrioStore(Protocol):
    """The corpus, as storage rather than as a constant.

    Versioning is only meaningful if something can write it: `superseded_by`
    was read in three places and writable from none until this existed.
    """

    def add(self, trio: Trio) -> Trio: ...

    def get(self, trio_id: str) -> Trio | None: ...

    def live(self) -> list[Trio]: ...

    def supersede(self, *, old_id: str, new_id: str) -> Trio: ...

    def seed(self, trios: Sequence[Trio]) -> None: ...


class InMemoryTrioStore:
    """Also the fallback when Postgres is unreachable: the agent still answers
    from the seed corpus, and edits simply do not survive the session."""

    def __init__(self, trios: Sequence[Trio] = ()) -> None:
        self._trios: dict[str, Trio] = {t.id: t for t in trios}

    def add(self, trio: Trio) -> Trio:
        self._trios[trio.id] = trio
        return trio

    def get(self, trio_id: str) -> Trio | None:
        return self._trios.get(trio_id)

    def live(self) -> list[Trio]:
        return [t for t in self._trios.values() if t.superseded_by is None]

    def supersede(self, *, old_id: str, new_id: str) -> Trio:
        old = self._trios.get(old_id)
        if old is None:
            raise KeyError(f"no trio {old_id!r}")
        replaced = replace(old, superseded_by=new_id)
        self._trios[old_id] = replaced
        return replaced

    def seed(self, trios: Sequence[Trio]) -> None:
        for trio in trios:
            self._trios.setdefault(trio.id, trio)


def live_trios(store) -> list[Trio]:
    """The corpus for this turn, whatever shape the caller holds.

    Accepts a store or a plain list so a test can pass either, and never fails
    a turn: an unreachable corpus costs grounding, not the answer.
    """
    if store is None:
        return []
    if isinstance(store, (list, tuple)):
        return [t for t in store if t.superseded_by is None]
    try:
        return store.live()
    except Exception:
        return []


class PostgresTrioStore:
    """The corpus in Postgres. Superseding sets a column; nothing is deleted."""

    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def add(self, trio: Trio) -> Trio:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from retail_agent.store.models import TrioRow

        values = {
            "id": trio.id,
            "question": trio.question,
            "sql": trio.sql,
            "report": trio.report,
            "metric_definitions": dict(trio.metric_definitions),
            "tags": list(trio.tags),
            "author": trio.author,
            "version": trio.version,
            "superseded_by": trio.superseded_by,
            "approved_at": trio.approved_at,
        }
        with self._sessions.begin() as session:
            session.execute(
                pg_insert(TrioRow)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={k: v for k, v in values.items() if k != "id"},
                )
            )
        return trio

    def get(self, trio_id: str) -> Trio | None:
        from sqlalchemy import select

        from retail_agent.store.models import TrioRow

        with self._sessions() as session:
            row = session.scalar(select(TrioRow).where(TrioRow.id == trio_id))
        return _to_trio(row) if row else None

    def live(self) -> list[Trio]:
        from sqlalchemy import select

        from retail_agent.store.models import TrioRow

        with self._sessions() as session:
            rows = session.scalars(
                select(TrioRow).where(TrioRow.superseded_by.is_(None)).order_by(TrioRow.id)
            ).all()
        return [_to_trio(row) for row in rows]

    def supersede(self, *, old_id: str, new_id: str) -> Trio:
        from sqlalchemy import update

        from retail_agent.store.models import TrioRow

        if self.get(old_id) is None:
            raise KeyError(f"no trio {old_id!r}")
        with self._sessions.begin() as session:
            session.execute(
                update(TrioRow).where(TrioRow.id == old_id).values(superseded_by=new_id)
            )
        return self.get(old_id)

    def seed(self, trios: Sequence[Trio]) -> None:
        """Insert what is absent, leave what is there. An analyst's edit has to
        survive a restart."""
        existing = {t.id for t in self._all()}
        for trio in trios:
            if trio.id not in existing:
                self.add(trio)

    def _all(self) -> list[Trio]:
        from sqlalchemy import select

        from retail_agent.store.models import TrioRow

        with self._sessions() as session:
            return [_to_trio(r) for r in session.scalars(select(TrioRow)).all()]


def _to_trio(row) -> Trio:
    return Trio(
        id=row.id,
        question=row.question,
        sql=row.sql,
        report=row.report,
        metric_definitions=dict(row.metric_definitions or {}),
        tags=tuple(row.tags or ()),
        author=row.author,
        version=row.version,
        superseded_by=row.superseded_by,
        approved_at=row.approved_at,
    )


def build_trio_store(settings, on_degraded=None):
    """Postgres when reachable, memory when not — seeded either way, so the
    agent always has the analysts' definitions even with no database."""
    import logging

    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.store.db import create_db_engine, session_factory

    try:
        engine = create_db_engine(settings.database_url)
        with engine.connect():
            pass
        store = PostgresTrioStore(session_factory(engine))
        store.seed(SEED_TRIOS)
        return store
    except Exception as err:
        logging.getLogger(__name__).debug("trio store degraded: %s", err)
        if on_degraded is not None:
            on_degraded()
        return InMemoryTrioStore(SEED_TRIOS)
