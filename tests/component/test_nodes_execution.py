import pandas as pd

from retail_agent.agent.nodes.execute import execute_node
from retail_agent.agent.nodes.sql import draft_sql_node
from retail_agent.agent.nodes.synthesize import synthesize_node
from retail_agent.agent.state import AnalysisStep, MaskedFrame, new_turn_state
from tests.component.conftest import FakeSource


def analysing_state(question="top customers", sql=None):
    state = new_turn_state(
        user_id="dana", session_id="s1", question=question, repair_budget=2
    )
    state["plan"] = [AnalysisStep(id="step_1", question=question, sql=sql)]
    state["step_index"] = 0
    return state


def test_draft_sql_strips_markdown_fences(make_deps):
    deps = make_deps(["```sql\nSELECT id FROM users LIMIT 5\n```"])
    result = draft_sql_node(analysing_state(), deps)

    assert result["plan"][0].sql.startswith("SELECT")
    assert "```" not in result["plan"][0].sql


def test_draft_sql_records_a_guard_violation_without_executing(make_deps, source):
    deps = make_deps(["SELECT email FROM users"], src=source)
    result = draft_sql_node(analysing_state(), deps)

    attempt = result["sql_attempts"][-1]
    assert attempt.violations
    assert source.executed == []


def test_guard_violation_spends_repair_budget(make_deps):
    deps = make_deps(["SELECT email FROM users"])
    result = draft_sql_node(analysing_state(), deps)

    assert result["repair_budget"] == 1


def test_guard_injects_a_limit_into_accepted_sql(make_deps):
    deps = make_deps(["SELECT id FROM users"])
    result = draft_sql_node(analysing_state(), deps)

    assert "LIMIT" in result["plan"][0].sql.upper()


def test_repair_prompt_includes_the_previous_error(make_deps):
    deps = make_deps(["SELECT email FROM users", "SELECT id FROM users"])
    state = analysing_state()

    first = draft_sql_node(state, deps)
    state.update(first)
    draft_sql_node(state, deps)

    assert "This query failed" in deps.llm.prompts[-1]
    assert "email" in deps.llm.prompts[-1]


def test_execute_masks_pii_before_storing_the_frame(make_deps, source):
    deps = make_deps([], src=source)
    state = analysing_state(sql="SELECT id, email FROM users LIMIT 2")

    result = execute_node(state, deps)

    frame = result["frames"]["step_1"]
    assert "@" not in str(frame.frame["email"].iloc[0])
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
        "step_1": MaskedFrame(
            key="step_1",
            frame=pd.DataFrame({"id": [1], "spend": [900]}),
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
        SqlAttempt(step_id="step_1", sql="SELECT email FROM users", violations=("no PII",))
    ]

    result = synthesize_node(state, deps)

    assert "SELECT email FROM users" in result["answer"]
    assert "no PII" in result["answer"]
