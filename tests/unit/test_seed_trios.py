"""The seed corpus is the system's ground truth. If it is wrong or
self-contradictory, every answer built on it is confidently wrong.
"""

import pytest

from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.knowledge.trios import UNDEFINED_TERMS, undefined_terms, unresolved


def test_every_trio_is_complete():
    for trio in SEED_TRIOS:
        assert trio.question and trio.sql and trio.report, trio.id
        assert trio.metric_definitions, f"{trio.id} defines nothing, so it teaches nothing"
        assert trio.tags, trio.id


def test_ids_are_unique():
    ids = [t.id for t in SEED_TRIOS]

    assert len(ids) == len(set(ids))


def test_every_defined_term_is_one_the_agent_actually_flags():
    """A definition for a term nothing detects can never be applied. This is
    the seam where the corpus and the detector silently drift apart."""
    defined = {term.lower() for t in SEED_TRIOS for term in t.metric_definitions}
    detectable = set(UNDEFINED_TERMS)

    orphans = {
        term for term in defined
        if term not in detectable and term not in {"margin"}
    }

    assert not orphans, f"defined but never detected: {orphans}"


@pytest.mark.parametrize(
    "question",
    [
        "why did our churn rate spike last month?",
        "who are our top customers?",
        "why are users in state X underspending?",
        "which brands are performing well?",
    ],
)
def test_the_briefs_questions_are_answered_by_the_corpus(question):
    """These are the brief's own examples. Every one raises a term the schema
    cannot settle, and the corpus exists to settle them."""
    assert undefined_terms(question), "the question should raise a term"

    found = retrieve(question, list(SEED_TRIOS))

    assert found, f"nothing retrieved for {question!r}"
    assert unresolved(question, found) == [], f"{question!r} left a term undefined"


def test_the_sql_in_each_trio_passes_the_safety_guard():
    """A trio is an example the agent may follow. One carrying SQL the guard
    would reject teaches it to write queries that cannot run."""
    from retail_agent.config import Settings
    from retail_agent.safety.pii import PiiPolicy
    from retail_agent.safety.sql_guard import check_sql

    settings = Settings(_env_file=None)
    policy = PiiPolicy.default()

    for trio in SEED_TRIOS:
        verdict = check_sql(
            trio.sql,
            allowed_tables=settings.allowed_tables,
            restricted_columns=policy.restricted_columns(),
            default_limit=settings.max_row_limit,
            max_limit=settings.max_row_limit,
        )
        assert verdict.ok, f"{trio.id}: {verdict.violations}"


def test_definitions_exclude_cancelled_and_returned_where_money_is_counted():
    """The single most repeated mistake against theLook. A corpus that gets it
    wrong teaches it to every future answer."""
    for trio in SEED_TRIOS:
        if "sale_price" in trio.sql:
            assert "Cancelled" in trio.sql and "Returned" in trio.sql, trio.id
