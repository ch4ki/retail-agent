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
            "SELECT email FROM users",  # rejected by the guard
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
            "SELECT email FROM users",
            "SELECT email FROM users",
            "SELECT email FROM users",
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

    stored = state["frames"]["step_1"].frame
    assert "@" not in str(stored["email"].iloc[0])
    assert state["redactions"] == 2


def test_guard_blocks_a_bare_pii_projection_end_to_end(make_deps, source):
    deps = make_deps(
        [
            "analyze",
            "STEP: customer emails",
            "SELECT id, email FROM users",
            "SELECT id, email FROM users",
            "SELECT id, email FROM users",
        ],
        src=source,
    )
    state = turn(build_graph(deps), "list every customer email")

    assert source.executed == [], "a PII query must never reach the warehouse"
    assert state["status"] == "degraded"
