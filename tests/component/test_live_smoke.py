"""Live checks against real BigQuery. Deselected by default.

Run with: uv run pytest -m live
"""

import pytest

from retail_agent.config import Settings
from retail_agent.datasources.bigquery import BigQuerySource

pytestmark = pytest.mark.live


@pytest.fixture
def source():
    # GOOGLE_CLOUD_PROJECT is an explicit override. When it is unset the client
    # falls back to the project in application-default credentials, so skip on
    # a real connection failure rather than on a missing env var.
    try:
        return BigQuerySource(Settings())
    except Exception as err:
        pytest.skip(f"BigQuery unavailable: {err}")


def test_describe_users_returns_real_columns(source):
    schema = source.describe("users")
    assert "email" in schema.column_names()


def test_small_query_returns_a_dataframe(source):
    result = source.execute(
        "SELECT id, state FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 5"
    )
    assert result.row_count == 5
    assert list(result.rows.columns) == ["id", "state"]


def test_dry_run_reports_bytes(source):
    estimate = source.dry_run(
        "SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`"
    )
    assert estimate.bytes_processed > 0


def test_bad_column_is_classified_as_a_syntax_error(source):
    # This is the failure the repair edge keys off, so the classification of a
    # real BigQuery 400 matters more than the message text.
    from retail_agent.datasources.base import QuerySyntaxError

    with pytest.raises(QuerySyntaxError):
        source.execute(
            "SELECT nope FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 1"
        )


# Structured output varies more across providers than plain text does
# (json_schema vs function_calling vs json_mode), and the component tests use a
# double that validates rather than negotiates. These call the real provider.


def _llm():
    from retail_agent.config import Settings
    from retail_agent.llm.provider import build_llm

    try:
        return build_llm(Settings())
    except Exception as err:
        pytest.skip(f"LLM unavailable: {err}")


def test_router_schema_works_against_the_real_provider():
    from retail_agent.agent.nodes.route import RouteDecision

    decision = _llm().with_structured_output(RouteDecision).invoke(
        "Which category is this: 'what tables do you have?'"
    )

    assert decision.intent in {"schema", "analyze", "chat"}


def test_planner_schema_works_against_the_real_provider():
    from retail_agent.agent.nodes.plan import Plan

    plan = _llm().with_structured_output(Plan).invoke(
        "Break into retrieval steps: compare revenue for brand X and brand Y."
    )

    assert isinstance(plan.steps, list)
    assert all(isinstance(step, str) for step in plan.steps)
    assert plan.steps, "a comparison should decompose into at least one step"


# Dense retrieval against the real embedding model. Marked live because the
# first run downloads it; the unit tests inject a fake embedder instead.


def test_dense_retrieval_finds_a_synonym_the_lexical_ranker_misses():
    """The case that justifies hybrid search: the executive says "lapsed", the
    analyst wrote "churned", and no word overlaps."""
    from retail_agent.knowledge.dense import MilvusDenseIndex
    from retail_agent.knowledge.retrieval import retrieve
    from retail_agent.knowledge.seeds import SEED_TRIOS

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        index = MilvusDenseIndex(path=f"{tmp}/trios.db")
        question = "how many shoppers have gone quiet?"

        lexical_only = retrieve(question, list(SEED_TRIOS))
        hybrid = retrieve(question, list(SEED_TRIOS), dense_rank=index.rank)

    assert not lexical_only, "no shared vocabulary with any trio"
    assert hybrid, "the embedding model should recognise the paraphrase"
