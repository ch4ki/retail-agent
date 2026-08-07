"""The eval corpus itself.

A corpus of hand-written SQL is a place for silent mistakes: a reference query
that is subtly wrong makes a correct agent look broken, and nobody suspects the
ruler. These assert what can be checked without a warehouse.
"""

from __future__ import annotations

import pytest

from retail_agent.config import Settings
from retail_agent.evals.cases import EVAL_CASES
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.safety.pii import PiiPolicy
from retail_agent.safety.sql_guard import check_sql


def test_the_corpus_is_big_enough_to_mean_something():
    """The design called for about forty. A handful of cases produces an
    accuracy figure that swings 20 points on one flaky turn."""
    assert len(EVAL_CASES) >= 40


def test_case_ids_are_unique():
    ids = [case.id for case in EVAL_CASES]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.id)
def test_every_reference_query_passes_the_same_guard_the_agent_faces(case):
    """The reference SQL is held to the project's own safety rules. A reference
    query that selects PII would leak it into the eval report — which is stored,
    printed and pasted into pull requests."""
    settings = Settings(_env_file=None)

    verdict = check_sql(
        case.reference_sql,
        allowed_tables=settings.allowed_tables,
        restricted_columns=PiiPolicy.default().restricted_columns(),
        default_limit=settings.max_row_limit,
        max_limit=settings.max_row_limit,
        qualify_with=settings.bq_dataset,
    )

    assert verdict.ok, f"{case.id}: {verdict.violations}"


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.id)
def test_every_case_asks_something(case):
    assert case.question.strip()
    assert case.reference_sql.strip()


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.id)
def test_required_definitions_exist_in_the_corpus(case):
    """A case that depends on "loyal" is measuring the agent's guess unless the
    Golden Bucket actually defines it. Those cases exist on purpose, but they
    have to be labelled rather than accidental."""
    defined = {
        term.lower()
        for trio in SEED_TRIOS
        for term in trio.metric_definitions
    }

    for term in case.required_definitions:
        assert term.lower() in defined, (
            f"{case.id} requires '{term}', which no seed trio defines"
        )


def test_the_corpus_covers_the_definition_dependent_questions():
    """These are the ones that motivated the suite. Without them it measures
    arithmetic the agent was never going to get wrong."""
    required = {term for case in EVAL_CASES for term in case.required_definitions}

    assert {"loyal", "churn", "top", "underspending"} <= required


def test_ranked_cases_exist():
    """Scalar-only coverage would miss ordering bugs entirely."""
    assert any(case.ranked for case in EVAL_CASES)
