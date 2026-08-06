import pandas as pd

from retail_agent.agent.state import (
    MaskedFrame,
    SqlAttempt,
    TurnState,
    fresh_scratch,
    new_turn_state,
)

# Keys that belong to the conversation, not to one turn of it.
DURABLE = {"user_id", "session_id", "messages"}


def test_every_turn_state_field_is_classified():
    """Add a field to TurnState and this fails until it is declared durable or
    scratch. A field that is neither is one the next turn silently inherits —
    which is how a spent repair budget and a stale result frame both got
    through."""
    scratch = set(fresh_scratch(repair_budget=2))

    assert set(TurnState.__annotations__) == DURABLE | scratch


def test_durable_and_scratch_do_not_overlap():
    """`messages` in the scratch set would wipe the conversation every turn."""
    assert DURABLE & set(fresh_scratch(repair_budget=2)) == set()


def test_destructive_action_fields_are_per_turn_scratch():
    """A resolved delete surviving into the next turn would let an unrelated
    "y" execute a deletion the user has stopped thinking about."""
    scratch = fresh_scratch(repair_budget=3)

    assert scratch["pending_action"] is None
    assert scratch["confirmation"] == ""


def test_scratch_carries_the_configured_repair_budget():
    assert fresh_scratch(repair_budget=3)["repair_budget"] == 3


def test_each_turn_gets_a_unique_id():
    assert fresh_scratch(repair_budget=2)["turn_id"] != fresh_scratch(
        repair_budget=2
    )["turn_id"]


def test_new_turn_state_seeds_only_the_conversation():
    """Scratch is the graph's job now, so the CLI entry point seeds identity and
    the question and nothing else."""
    state = new_turn_state(user_id="dana", session_id="s1", question="top customers?")

    assert state["user_id"] == "dana"
    assert state["messages"][0].content == "top customers?"
    assert set(state) == DURABLE


def test_masked_frame_renders_markdown():
    frame = MaskedFrame.from_dataframe(
        pd.DataFrame({"state": ["TX", "CA"], "revenue": [10, 20]}),
        row_count=2,
        redactions=0,
    )

    rendered = frame.to_markdown()

    assert "state" in rendered
    assert "TX" in rendered


def test_masked_frame_truncates_long_results():
    frame = MaskedFrame.from_dataframe(
        pd.DataFrame({"n": range(100)}),
        row_count=100,
        redactions=0,
    )

    rendered = frame.to_markdown(max_rows=5)

    assert "95 more rows" in rendered


def test_attempt_with_violations_counts_as_failed():
    assert SqlAttempt(step_id="s", sql="x", violations=("bad",)).failed


def test_attempt_with_an_error_counts_as_failed():
    assert SqlAttempt(step_id="s", sql="x", error="boom").failed


def test_successful_attempt_is_not_failed():
    assert not SqlAttempt(step_id="s", sql="x", row_count=3).failed
