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


# Dense retrieval against the real embedding model and a real database. Marked
# live because it bills OpenAI per run; the component tests inject a fake
# embedder and assert the storage instead.


def test_dense_retrieval_finds_a_synonym_the_lexical_ranker_misses():
    """The case that justifies hybrid search: the executive says "gone quiet",
    the analyst wrote "churned", and no word overlaps."""
    import pytest

    from retail_agent.config import Settings
    from retail_agent.knowledge.dense import build_dense_index
    from retail_agent.knowledge.retrieval import retrieve
    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.knowledge.trios import PostgresTrioStore
    from retail_agent.store.db import create_db_engine, run_migrations, session_factory

    settings = Settings(dense_retrieval=True)
    if not settings.openai_api_key:
        pytest.skip("no OPENAI_API_KEY configured")

    try:
        run_migrations(settings.database_url)
        engine = create_db_engine(settings.database_url)
    except Exception as err:
        pytest.skip(f"Postgres unavailable: {err}")

    try:
        sessions = session_factory(engine)
        PostgresTrioStore(sessions).seed(SEED_TRIOS)
        index = build_dense_index(settings, sessions=sessions)
        assert index is not None

        question = "how many shoppers have gone quiet?"
        lexical_only = retrieve(question, list(SEED_TRIOS))
        hybrid = retrieve(question, list(SEED_TRIOS), dense_rank=index.rank)
    finally:
        engine.dispose()

    assert not lexical_only, "no shared vocabulary with any trio"
    assert hybrid, "the embedding model should recognise the paraphrase"


# The non-regression that the regex detector was written conservatively to
# protect, now that a model makes the call instead. These are the brief's own
# example questions: reading them as "this user likes long answers" would have
# the agent quietly deciding every analyst wants essays.


def test_causal_questions_are_not_read_as_style_preferences():
    from retail_agent.agent.nodes.route import RouteDecision, style_signal
    from retail_agent.agent.prompts import ROUTER_PROMPT
    from retail_agent.config import get_settings
    from retail_agent.llm.provider import build_llm

    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm(get_settings()).with_structured_output(RouteDecision)
    prompt = ROUTER_PROMPT.format(history="(first question)")

    misread = []
    for question in [
        "why are users in state X underspending?",
        "why did our churn rate spike last month?",
        "why does brand Calvin Klein outperform brand Levis?",
        "explain the drop in March",
        "how come revenue fell?",
        "what was revenue in March?",
        "who are our top 10 customers by spend?",
    ]:
        decision = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)]
        )
        if style_signal(decision, question=question) is not None:
            misread.append((question, decision.style_field, decision.style_value))

    assert not misread, f"read as style preferences: {misread}"


def test_a_real_preference_is_detected_with_a_quotable_span():
    """The half the regex could not do: no pattern in the old list matches any
    of these, and the negation case recorded the opposite."""
    from retail_agent.agent.nodes.route import RouteDecision, style_signal
    from retail_agent.agent.prompts import ROUTER_PROMPT
    from retail_agent.config import get_settings
    from retail_agent.llm.provider import build_llm

    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_llm(get_settings()).with_structured_output(RouteDecision)
    prompt = ROUTER_PROMPT.format(history="(first question)")

    missed = []
    for question, expected in [
        ("cut to the chase, how many brands?", "summary"),
        ("spare me the details — what was Q1 revenue?", "summary"),
        ("walk me through it: why did churn spike?", "deep"),
        ("don't just give me the number, tell me why", "deep"),
    ]:
        decision = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)]
        )
        signal = style_signal(decision, question=question)
        if signal is None or signal.value != expected:
            missed.append((question, expected, signal))

    assert not missed, f"not detected as expected: {missed}"
