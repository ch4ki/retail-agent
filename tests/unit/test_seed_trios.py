"""The seed corpus is the system's ground truth. If it is wrong or
self-contradictory, every answer built on it is confidently wrong.
"""

import pytest

from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.seeds import SEED_TRIOS


def test_every_trio_is_complete():
    for trio in SEED_TRIOS:
        assert trio.question and trio.sql and trio.report, trio.id
        assert trio.metric_definitions, f"{trio.id} defines nothing, so it teaches nothing"
        assert trio.tags, trio.id


def test_ids_are_unique():
    ids = [t.id for t in SEED_TRIOS]

    assert len(ids) == len(set(ids))


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
    """These are the brief's own examples. Every one turns on a term the schema
    cannot settle, and the corpus exists to settle them.

    What this can no longer assert is that a detector flags the term — there is
    no detector, and the test that checked the corpus and the word list had not
    drifted apart went with it. What still matters, and is what actually
    reaches the model, is that retrieval finds a trio and the trio brings a
    definition with it.
    """
    found = retrieve(question, list(SEED_TRIOS))

    assert found, f"nothing retrieved for {question!r}"
    assert any(t.metric_definitions for t in found), (
        f"{question!r} retrieved nothing that defines anything"
    )


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


def test_definitions_state_business_rules_not_query_recipes():
    """A definition says what the business means; how to build the query is
    the analyst prompt's job. A recipe keyed to one retrievable term teaches
    the lesson only when that exact term is asked — "engaged share" would hit
    the identical trap with no seed to save it — and it renders to the
    executive via /trios as though it defined something."""
    for trio in SEED_TRIOS:
        for term, meaning in trio.metric_definitions.items():
            assert "COUNTIF" not in meaning, (term, trio.id)
            assert "subquery" not in meaning.lower(), (term, trio.id)


def test_the_share_denominator_rule_reaches_every_share_question():
    """The numerator/denominator traps are a property of every share question,
    so the rule lives in the prompt read on every query, not in one seed."""
    from retail_agent.agent.prompts import ANALYST_PROMPT

    assert "COUNTIF" in ANALYST_PROMPT
    assert "share" in ANALYST_PROMPT.lower()
