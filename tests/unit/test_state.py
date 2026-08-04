import pandas as pd

from retail_agent.agent.state import MaskedFrame, SqlAttempt, new_turn_state


def test_new_turn_state_seeds_the_question_and_budget():
    state = new_turn_state(
        user_id="dana", session_id="s1", question="top customers?", repair_budget=2
    )

    assert state["user_id"] == "dana"
    assert state["repair_budget"] == 2
    assert state["messages"][0].content == "top customers?"
    assert state["status"] == "ok"
    assert state["plan"] == []


def test_each_turn_gets_a_unique_id():
    a = new_turn_state(user_id="d", session_id="s", question="q", repair_budget=2)
    b = new_turn_state(user_id="d", session_id="s", question="q", repair_budget=2)
    assert a["turn_id"] != b["turn_id"]


def test_masked_frame_renders_markdown():
    frame = MaskedFrame(
        key="step_1",
        frame=pd.DataFrame({"state": ["TX", "CA"], "revenue": [10, 20]}),
        row_count=2,
        redactions=0,
        dropped_columns=(),
    )

    rendered = frame.to_markdown()

    assert "state" in rendered
    assert "TX" in rendered


def test_masked_frame_truncates_long_results():
    frame = MaskedFrame(
        key="step_1",
        frame=pd.DataFrame({"n": range(100)}),
        row_count=100,
        redactions=0,
        dropped_columns=(),
    )

    rendered = frame.to_markdown(max_rows=5)

    assert "95 more rows" in rendered


def test_attempt_with_violations_counts_as_failed():
    assert SqlAttempt(step_id="s", sql="x", violations=("bad",)).failed


def test_attempt_with_an_error_counts_as_failed():
    assert SqlAttempt(step_id="s", sql="x", error="boom").failed


def test_successful_attempt_is_not_failed():
    assert not SqlAttempt(step_id="s", sql="x", row_count=3).failed
