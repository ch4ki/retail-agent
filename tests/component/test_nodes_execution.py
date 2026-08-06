import pandas as pd

from retail_agent.agent.nodes.execute import execute_node
from retail_agent.agent.nodes.sql import draft_sql_node
from retail_agent.agent.nodes.synthesize import synthesize_node
from retail_agent.agent.state import (
    AnalysisStep,
    MaskedFrame,
    SqlAttempt,
    fresh_scratch,
    new_turn_state,
)
from tests.component.conftest import FakeSource


def analysing_state(question="top customers", sql=None):
    # The state as `start_turn` leaves it, since these call nodes directly.
    state = new_turn_state(user_id="dana", session_id="s1", question=question)
    state.update(fresh_scratch(repair_budget=2))
    state["plan"] = [AnalysisStep(id="step_1", question=question, sql=sql)]
    return state


def test_draft_sql_strips_markdown_fences(make_deps):
    deps = make_deps(["```sql\nSELECT id FROM users LIMIT 5\n```"])
    result = draft_sql_node(analysing_state(), deps)

    assert result["plan"][0].sql.startswith("SELECT")
    assert "```" not in result["plan"][0].sql


def test_draft_sql_records_a_guard_violation_without_executing(make_deps, source):
    deps = make_deps(["SELECT email AS contact FROM users"], src=source)
    result = draft_sql_node(analysing_state(), deps)

    attempt = result["sql_attempts"][-1]
    assert attempt.violations
    assert source.executed == []


def test_guard_violation_spends_repair_budget(make_deps):
    deps = make_deps(["SELECT email AS contact FROM users"])
    result = draft_sql_node(analysing_state(), deps)

    assert result["repair_budget"] == 1


def test_guard_injects_a_limit_into_accepted_sql(make_deps):
    deps = make_deps(["SELECT id FROM users"])
    result = draft_sql_node(analysing_state(), deps)

    assert "LIMIT" in result["plan"][0].sql.upper()


def test_repair_prompt_includes_the_previous_error(make_deps):
    deps = make_deps(["SELECT email AS contact FROM users", "SELECT id FROM users"])
    state = analysing_state()

    first = draft_sql_node(state, deps)
    state.update(first)
    draft_sql_node(state, deps)

    assert "This query failed" in deps.llm.prompts[-1]
    assert "email" in deps.llm.prompts[-1]


def test_repair_prompt_shows_the_sql_that_actually_ran(make_deps):
    """The warehouse error describes the query the guard rewrote, not the one
    the model typed. Pairing that error with the pre-rewrite SQL asks the model
    to fix a query it never ran."""
    broken = FakeSource(frames={}, failing={"BROKEN"})
    deps = make_deps(["SELECT id FROM users"], src=broken)
    state = analysing_state()
    state["plan"][0].sql = "SELECT BROKEN FROM `bigquery-public-data.thelook.users`"
    state["sql_attempts"] = [SqlAttempt(step_id="step_1", sql="SELECT BROKEN FROM users")]

    state.update(execute_node(state, deps))
    draft_sql_node(state, deps)

    assert "`bigquery-public-data.thelook.users`" in deps.llm.prompts[-1]


def test_execute_masks_pii_before_storing_the_frame(make_deps, source):
    deps = make_deps([], src=source)
    state = analysing_state(sql="SELECT id, email FROM users LIMIT 2")

    result = execute_node(state, deps)

    frame = result["frames"]["step_1"]
    assert "@" not in str(frame.column("email")[0])
    assert result["redactions"] > 0


def test_execute_records_a_syntax_error_and_keeps_going(make_deps):
    broken = FakeSource(frames={}, failing={"BROKEN"})
    deps = make_deps([], src=broken)
    state = analysing_state(sql="SELECT BROKEN FROM users")

    result = execute_node(state, deps)

    assert result["sql_attempts"][-1].error is not None
    assert result["repair_budget"] == 1


def test_execute_advances_the_step_index_on_success(make_deps):
    deps = make_deps([])
    result = execute_node(analysing_state(sql="SELECT id FROM users LIMIT 1"), deps)

    assert result["step_index"] == 1


def test_synthesize_scans_output_for_pii(make_deps):
    deps = make_deps(["Our top customer is ada@example.com with $900."])
    state = analysing_state()
    state["frames"] = {
        "step_1": MaskedFrame.from_dataframe(
            pd.DataFrame({"id": [1], "spend": [900]}),
            row_count=1,
            redactions=0,
        )
    }

    result = synthesize_node(state, deps)

    assert "ada@example.com" not in result["answer"]
    assert "[redacted:email]" in result["answer"]


def test_synthesize_reports_honestly_when_every_step_failed(make_deps):
    deps = make_deps([])
    state = analysing_state()
    state["frames"] = {}
    state["status"] = "degraded"

    result = synthesize_node(state, deps)

    answer = result["answer"].lower()
    assert "couldn't" in answer or "could not" in answer


def test_synthesize_failure_message_shows_the_last_attempt(make_deps):
    from retail_agent.agent.state import SqlAttempt

    deps = make_deps([])
    state = analysing_state()
    state["frames"] = {}
    state["sql_attempts"] = [
        SqlAttempt(step_id="step_1", sql="SELECT email AS contact FROM users", violations=("no PII",))
    ]

    result = synthesize_node(state, deps)

    assert "SELECT email AS contact FROM users" in result["answer"]
    assert "no PII" in result["answer"]


def test_partial_step_failure_is_reported_not_glossed_over(make_deps):
    """If the plan had 3 steps and only 1 produced data, saying 'there is no
    data' is wrong — the query failed. The turn must be marked degraded and the
    gap named."""
    import pandas as pd

    from retail_agent.agent.state import AnalysisStep, SqlAttempt

    deps = make_deps(["Jeans made $106,915."])
    state = analysing_state()
    state["plan"] = [
        AnalysisStep(id="step_1", question="revenue for Jeans"),
        AnalysisStep(id="step_2", question="revenue for Sweaters"),
        AnalysisStep(id="step_3", question="units per category"),
    ]
    state["frames"] = {
        "step_1": MaskedFrame.from_dataframe(
            pd.DataFrame({"revenue": [106915]}), row_count=1, redactions=0
        )
    }
    state["sql_attempts"] = [
        SqlAttempt(step_id="step_2", sql="SELECT bad", error="boom"),
    ]

    result = synthesize_node(state, deps)

    assert result["status"] == "degraded"
    prompt = deps.llm.prompts[-1]
    assert "revenue for Sweaters" in prompt, "the model must know which step failed"
    assert "units per category" in prompt


def test_all_steps_succeeding_stays_ok(make_deps):
    import pandas as pd

    from retail_agent.agent.state import AnalysisStep

    deps = make_deps(["All good."])
    state = analysing_state()
    state["plan"] = [AnalysisStep(id="step_1", question="revenue")]
    state["frames"] = {
        "step_1": MaskedFrame.from_dataframe(
            pd.DataFrame({"revenue": [1]}), row_count=1, redactions=0
        )
    }

    assert synthesize_node(state, deps)["status"] == "ok"


def test_unqualified_tables_are_qualified_before_execution(make_deps, source):
    """A model that forgets the dataset prefix gets a BigQuery 400. The guard
    fixes it rather than spending a repair attempt on it."""
    deps = make_deps(["SELECT id FROM users"], src=source)
    result = draft_sql_node(analysing_state(), deps)

    sql = result["plan"][0].sql
    assert "thelook_ecommerce.users" in sql
    assert result["sql_attempts"][-1].violations == ()
