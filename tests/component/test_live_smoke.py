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


# One real question, end to end. The component tests script the model, so
# nothing else in the suite exercises tool calling against the actual provider —
# and tool-call formats differ by vendor in a way a validating double hides.


def _llm():
    from retail_agent.config import Settings
    from retail_agent.llm.provider import build_llm

    try:
        return build_llm(Settings())
    except Exception as err:
        pytest.skip(f"LLM unavailable: {err}")


def test_a_real_question_produces_a_number_from_a_real_query(source):
    """Asserts on the captured frame, not on the prose.

    A model that answers "revenue was strong" without querying would pass any
    text assertion. What must be true is that a query ran and returned a row.
    """
    from retail_agent.agent.seams import ask_once
    from retail_agent.bootstrap import build_deps
    from retail_agent.config import Settings

    deps = build_deps(Settings(), llm=_llm(), source=source)
    answer = ask_once(deps, "How many orders were placed in March 2023?")

    assert answer.sql, "no query ran"
    assert answer.rows, "the query returned nothing"
    assert answer.intent == "analyze"


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


def _tools_called(source, question, tool):
    """Run one real turn and report where `tool` fell among the steps taken.

    Detection used to be folded into the router's structured output, where it
    cost nothing and could not be skipped. It is a tool now, so whether the
    model elects to call it is a behaviour of the live provider — which means it
    can only be checked against the live provider.
    """
    from retail_agent.agent.capture import TurnCapture
    from retail_agent.agent.deps import TurnContext
    from retail_agent.agent.supervisor import build_agent
    from retail_agent.bootstrap import build_deps
    from retail_agent.config import Settings

    from langgraph.checkpoint.memory import MemorySaver

    deps = build_deps(Settings(), llm=_llm(), source=source)
    capture = TurnCapture(question=question)
    agent = build_agent(deps, capture, checkpointer=MemorySaver())
    agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": "live"}},
        context=TurnContext(user_id="live", session_id="live", turn_id="t1"),
    )
    return [name for name, _, _ in capture.events if name == tool]


def _note_preference_calls(source, question):
    return _tools_called(source, question, "note_preference")


def _step_order(source, question):
    """Every step in order, so a test can assert that one preceded another."""
    from retail_agent.agent.capture import TurnCapture
    from retail_agent.agent.deps import TurnContext
    from retail_agent.agent.supervisor import build_agent
    from retail_agent.bootstrap import build_deps
    from retail_agent.config import Settings

    from langgraph.checkpoint.memory import MemorySaver

    deps = build_deps(Settings(), llm=_llm(), source=source)
    capture = TurnCapture(question=question)
    agent = build_agent(deps, capture, checkpointer=MemorySaver())
    agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": "live"}},
        context=TurnContext(user_id="live", session_id="live", turn_id="t1"),
    )
    return [name for name, _, _ in capture.events]


@pytest.mark.parametrize(
    "question",
    [
        "why are users in state Texas underspending?",
        "why did our churn rate spike last month?",
        "what was revenue in March?",
    ],
)
def test_a_question_about_the_data_is_not_read_as_a_style_preference(source, question):
    """The failure mode the regex this replaced actually had.

    "don't just give me the number, tell me why" recorded `depth=summary` — the
    opposite of what was asked. A causal question misfiling a preference is the
    same class of error, and it silently changes how later answers are written.
    """
    assert _note_preference_calls(source, question) == []


def test_a_stated_preference_is_noticed(source):
    """The other half. A tool nothing ever calls is a learning loop that does
    not run."""
    called = _note_preference_calls(
        source, "spare me the details from now on — what was Q1 revenue?"
    )

    assert called, "the model did not record a preference the user stated outright"


# The measurement that replaced a guarantee. A regex over nineteen words could
# not fail to fire; a tool can simply not be called, so whether the model calls
# it is a property of the live provider and can only be checked against one.
# `evals.runner.ask_rate` is the same question asked over the whole suite.


def test_an_in_house_term_makes_the_agent_ask_before_it_queries(source):
    """The bug this replaced the word list for.

    "LGB" is in no dictionary the agent ships with, which is the point: the
    detector it replaced could only recognise words somebody had added, and
    this question came back with a confident number built on an invented
    meaning. Asserting on the order, not the call — asking after the query has
    already spent the money on the guess.
    """
    from retail_agent.evals.runner import asked_before_querying

    steps = _step_order(source, "make me a report on 10 LGB customers")

    assert asked_before_querying(steps), (
        f"the agent never asked what LGB meant before querying: {steps}"
    )


@pytest.mark.parametrize(
    "question",
    [
        "what was total revenue in March 2024?",
        "how many orders shipped last week?",
        "what is the average order value by state?",
    ],
)
def test_a_question_the_columns_answer_is_not_interrupted(source, question):
    """The other direction, and the one that decides whether this survives
    contact with users. A gate that asks about "revenue" gets clicked through,
    and then it is not a gate at all."""
    assert _tools_called(source, question, "ask_for_definitions") == []


# Counting entities that must first pass a per-entity test. This is the shape
# the eval caught the agent getting wrong on every definition-dependent count.


@pytest.mark.parametrize(
    "question",
    [
        "How many loyal customers do we have?",
        "How many engaged customers do we have?",
    ],
)
def test_a_count_behind_a_per_entity_threshold_returns_one_row(source, question):
    """`SELECT COUNT(DISTINCT user_id) ... GROUP BY user_id HAVING ...` returns
    one row per customer, each holding 1, and never computes the total.

    Live, all three definition-dependent counting cases did exactly that. Two
    still scored correct, because the *number of rows* happened to be the
    answer and the truncation warning tells the agent to read it. The third
    asked for a percentage, where no row count can stand in, and had nothing to
    divide.

    So the assertion is on the query's shape rather than on the figure: one row
    holding the count, which is the only form that also supports a ratio.
    """
    from retail_agent.agent.seams import ask_once
    from retail_agent.bootstrap import build_deps
    from retail_agent.config import Settings

    deps = build_deps(Settings(), llm=_llm(), source=source)
    answer = ask_once(deps, question)

    assert answer.sql, "no query ran"
    assert answer.row_count == 1, (
        f"expected a single row holding the count, got {answer.row_count} "
        f"rows — the query groups by the column it is counting.\n{answer.sql}"
    )
