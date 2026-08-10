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
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9']+")

# Filler carries no signal about which trio is relevant. "many" was in a live
# question — "how many shoppers have gone quiet?" — and matched a trio whose
# question read "How many loyal customers do we have?", on that word alone.
_STOPWORDS = frozenset(
    """a an and are as at be by did do does for from has have how in into is it
    its of on or our that the their they this to was were what when where which
    who why with you your me my show give tell
    many much more most some any all every each few lot lots number count
    there here we us i he she them him her it's dont don't can could would
    should will shall may might must been being am get got had having
    please just now then than so very really quite rather also too only
    over under about across between within during while before after""".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


# Words that reverse or qualify a term rather than ride along with it. Kept out
# of the filler set explicitly, whatever `_STOPWORDS` grows to include later:
# "not loyal" settled from `loyal` answers the opposite question, silently.
_NEVER_FILLER = frozenset(
    "not no never non without least fewest lowest worst former ex stopped".split()
)

# The subjects a business term describes. "loyal customers" must settle from
# `loyal`; these are the words allowed alongside a matched term without leaving
# the phrase open. Any word in neither set fails closed — the phrase stays
# open and gets asked about — so a qualifier missing from `_NEVER_FILLER`
# costs a question, never a wrong cohort.
_SUBJECTS = frozenset(
    "customer customers user users shopper shoppers buyer buyers client clients".split()
)

_FILLER = (_STOPWORDS | _SUBJECTS) - _NEVER_FILLER

# There used to be a dict of nineteen words here, matched by regex, and it was
# how the agent decided a question needed a definition before it could be
# answered. It could only ever recognise a word somebody had thought of in
# advance: "make me a report on 10 LGB customers" raised nothing and came back
# with a confident number. The model asks now, by calling `ask_for_definitions`
# — see the spec in
# `docs/superpowers/specs/2026-08-09-model-driven-term-detection-design.md`.


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


def agreed_definitions(trios: list[Trio]) -> dict[str, str]:
    """Every term these trios settle, as lowercased term → meaning.

    Earlier trios win, because `retrieve` returns them in relevance order and
    the best match should not be overwritten by an also-ran that happens to use
    the same word.

    Separate from `definitions_block` because two callers need different
    things from the same fact. The analyst needs it rendered into a prompt;
    `ask_for_definitions` needs to *look a term up* before telling the
    executive nobody has defined it.
    """
    seen: dict[str, str] = {}
    for trio in trios:
        for term, meaning in trio.metric_definitions.items():
            seen.setdefault(term.lower(), meaning)
    return seen


def lookup_definition(definitions: dict[str, str], phrase: str) -> str | None:
    """The definition of `phrase`, allowing for how the two sides name things.

    The model passes the executive's own words — its tool description tells it
    to — so it asks about "loyal customers", with whatever punctuation they
    typed. Definitions are keyed on the business term alone: `loyal`. Exact
    lookup misses, and the executive is told nobody has defined a term the
    analytics team agreed months ago.

    Matching is on whole words, never substrings: `disloyal customers` must not
    be settled by the definition of `loyal`. The most specific defined term
    wins, so "share of loyal customers" answers from `loyal share` rather than
    falling through to plain `loyal` and losing the rule the longer term
    carries. A phrase holding two independent terms keeps both meanings —
    returning one would drop a constraint, and *which* one would depend on
    dict order.

    Every word the match does not account for must be filler. A word that is
    neither part of a matched term nor in `_FILLER` — "not", "least", "semi",
    anything nobody anticipated — leaves the phrase open. That is the safe
    direction: an open phrase costs a question, a phrase settled by the
    positive half of "not loyal" costs a silently inverted cohort.
    """
    words = _WORD.findall(phrase.lower())
    if not words:
        return None

    present = set(words)
    covered: set[str] = set()
    matched: list[tuple[str, str]] = []
    for term in sorted(definitions, key=lambda t: (-len(tokenize(t)), t)):
        target = set(tokenize(term))
        if target and target <= present and not target <= covered:
            covered |= target
            matched.append((term, definitions[term]))

    if not matched:
        # A term made only of filler can never be token-covered; the stored
        # key itself is the one remaining way to find it.
        return definitions.get(phrase.lower().strip())
    if any(w not in covered and w not in _FILLER for w in words):
        return None
    if len(matched) == 1:
        return matched[0][1]
    return "; ".join(f"{term}: {meaning}" for term, meaning in matched)


def definitions_block(trios: list[Trio], *, except_for: Collection[str] = ()) -> str:
    """The definitions to put in front of the model, deduplicated.

    Only definitions — not the past SQL. A previous query carries its own date
    filters and joins, and pasting it into a new question is how an agent
    answers last quarter's question with this quarter's label.

    `except_for` withholds terms the executive has overridden with their own
    definition: rendering both meanings hands the model a choice it must not
    have.
    """
    overridden = {term.lower() for term in except_for}
    seen = {
        term: meaning
        for term, meaning in agreed_definitions(trios).items()
        if term not in overridden
    }

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

    The term alone, with no gloss on what was undecided. The terms are the
    executive's own words now rather than keys of a dict that shipped a
    description alongside each one, and the sentence this asks for — the rule
    the query actually applied — says more than the gloss ever did.
    """
    if not terms:
        return ""
    described = "; ".join(f"**{term}**" for term in terms)
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
    described = ", ".join(terms)
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

    @property
    def sessions(self):
        """Shared with the dense index, which keeps its vectors in the same
        database — one engine and one pool rather than two."""
        return self._sessions

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
    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.store.db import sessions_or_none

    sessions = sessions_or_none(
        settings.database_url, name="trio store", on_degraded=on_degraded
    )
    if sessions is None:
        return InMemoryTrioStore(SEED_TRIOS)

    store = PostgresTrioStore(sessions)
    store.seed(SEED_TRIOS)
    return store
