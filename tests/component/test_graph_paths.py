import pandas as pd

from retail_agent.agent.graph import build_graph, run_turn
from tests.component.conftest import FakeSource


def turn(graph, question="top customers by spend"):
    return run_turn(
        graph, user_id="dana", session_id="s1", question=question, repair_budget=2
    )


def test_schema_question_never_executes_sql(make_deps, source):
    deps = make_deps(["schema", "We hold orders, products and customers."], src=source)
    state = turn(build_graph(deps), "what data do you have?")

    assert source.executed == []
    assert "orders" in state["answer"]


def test_chat_follow_up_never_executes_sql(make_deps, source):
    deps = make_deps(["chat", "Glad that helped."], src=source)
    state = turn(build_graph(deps), "thanks, that's useful")

    assert source.executed == []
    assert state["answer"] == "Glad that helped."


def test_happy_path_produces_an_answer(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: total spend per customer",
            "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",
            "Your top customer spent $100.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert "$100" in state["answer"]
    assert len(source.executed) == 1


def test_guard_violation_triggers_one_repair_then_succeeds(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: top customers",
            "SELECT email AS contact FROM users",  # rejected by the guard
            "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",  # repaired
            "Your top customer spent $100.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert len(state["sql_attempts"]) >= 2
    assert source.executed, "the repaired query should have run"


def test_exhausted_repair_budget_degrades_instead_of_looping(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: top customers",
            "SELECT email AS contact FROM users",
            "SELECT email AS contact FROM users",
            "SELECT email AS contact FROM users",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "degraded"
    assert source.executed == []
    assert "couldn't build a working query" in state["answer"]


def test_syntax_error_is_repaired(make_deps):
    broken = FakeSource(
        frames={"ok": pd.DataFrame({"id": [1], "spend": [10]})}, failing={"BROKEN"}
    )
    deps = make_deps(
        [
            "analyze",
            "STEP: revenue",
            "SELECT BROKEN FROM users",
            "SELECT id, spend FROM users",
            "Revenue was $10.",
        ],
        src=broken,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert len(broken.executed) == 2


def test_multi_step_plan_runs_every_step(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: revenue for brand X\nSTEP: revenue for brand Y",
            "SELECT id, spend FROM users",
            "SELECT id, spend FROM users",
            "Brand X leads brand Y.",
        ],
        src=source,
    )
    state = turn(build_graph(deps), "compare brand X and brand Y")

    assert len(source.executed) == 2
    assert len(state["frames"]) == 2


# The warehouse returns an `email` column regardless of the SQL, so these two
# exercise masking at the data boundary rather than the guard's projection rule.


def test_pii_never_reaches_the_model_context(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: top customers",
            "SELECT id, spend FROM users",
            "Top customer identified.",
        ],
        src=source,
    )
    turn(build_graph(deps))

    synthesis_prompts = [p for p in deps.llm.prompts if "Query results" in p]
    assert synthesis_prompts, "synthesis should have been reached"
    assert "a@b.com" not in synthesis_prompts[-1]
    assert "@" not in synthesis_prompts[-1].split("Query results")[1]


def test_masked_frames_are_stored_not_raw_ones(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: top customers",
            "SELECT id, spend FROM users",
            "Done.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    stored = state["frames"]["step_1"]
    assert "@" not in str(stored.column("email")[0])
    assert state["redactions"] == 2


def test_guard_blocks_unmaskable_pii_projections_end_to_end(make_deps, source):
    # A live model produced the CONCAT(MAX(...)) form to work around the guard.
    # It renames the output column, so masking cannot find it — the query must
    # never reach the warehouse.
    evasion = (
        "SELECT u.id AS user_id, "
        "CONCAT(MAX(u.first_name), ' ', MAX(u.last_name)) AS user_name "
        "FROM users AS u GROUP BY user_id"
    )
    deps = make_deps(
        ["analyze", "STEP: customer names", evasion, evasion, evasion], src=source
    )
    state = turn(build_graph(deps), "list our top customers by name")

    assert source.executed == [], "an unmaskable PII query must never run"
    assert state["status"] == "degraded"


def test_bare_pii_column_runs_and_is_masked_end_to_end(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: customer contacts",
            "SELECT id, email FROM users",
            "Listed 2 customers.",
        ],
        src=source,
    )
    state = turn(build_graph(deps), "show me customer records")

    assert len(source.executed) == 1, "a maskable query is allowed to run"
    assert state["redactions"] == 2
    assert "@" not in str(state["frames"]["step_1"].column("email")[0])


def test_state_survives_checkpointing(make_deps, source):
    """The CLI always runs with a checkpointer, so every value in TurnState
    must be serialisable. Running without one hid a crash on the first
    analysis turn."""
    from langgraph.checkpoint.memory import MemorySaver

    deps = make_deps(
        [
            "analyze",
            "STEP: total spend per customer",
            "SELECT id, spend FROM users",
            "Your top customer spent $100.",
        ],
        src=source,
    )
    graph = build_graph(deps, checkpointer=MemorySaver())
    state = run_turn(
        graph,
        user_id="dana",
        session_id="s1",
        question="top customers",
        repair_budget=2,
        config={"configurable": {"thread_id": "s1"}},
    )

    assert state["status"] == "ok"
    assert state["frames"]["step_1"].row_count == 2
