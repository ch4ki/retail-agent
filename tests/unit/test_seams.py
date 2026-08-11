"""Reducing a finished turn to something an eval can score.

This is a pure function on purpose. It is the one part of the eval wiring that
decides what "the agent's answer" means, and a rule that could only be exercised
by a live run against BigQuery would never be checked at all.
"""

from retail_agent.agent.seams import answer_from_state
from retail_agent.agent.state import attempt_record, frame_to_state, step_event
from retail_agent.safety.frame import MaskedFrame


def frame(**kwargs) -> MaskedFrame:
    defaults = dict(columns=("n",), rows=((5_823,),), row_count=5_823, redactions=0)
    return MaskedFrame(**{**defaults, **kwargs})


def state_with(**overrides) -> dict:
    base = {
        "attempts": [],
        "events": [],
        "trio_ids": [],
        "calls": 0,
        "frame": None,
        "executed_sql": "",
        "reports_written": [],
    }
    base.update(overrides)
    return base


def test_the_number_comes_from_the_frame_and_not_from_the_prose():
    """Parsing prose would measure how the model phrased itself, which varies
    legitimately between model versions, rather than what the query returned."""
    f = frame()
    state = state_with(
        attempts=[
            attempt_record(sql="SELECT COUNT(*)", executed_sql="SELECT COUNT(*)")
        ],
        frame=frame_to_state(f),
        executed_sql="SELECT COUNT(*)",
    )

    answer = answer_from_state("About six thousand, give or take.", state)

    assert answer.rows == [[5_823]]
    assert answer.row_count == 5_823
    assert answer.text == "About six thousand, give or take."


def test_a_turn_that_never_queried_scores_as_no_rows_rather_than_raising():
    """A refused or chat turn is a legitimate outcome, not a harness error."""
    answer = answer_from_state("I can only help with the retail data.", state_with())

    assert answer.rows == []
    assert answer.columns == ()
    assert answer.sql == ""
    assert answer.intent == "chat"


def test_the_scored_query_is_the_one_that_ran():
    """A repaired turn has several attempts and only the last produced a number."""
    state = state_with(
        attempts=[
            attempt_record(sql="SELECT bad", error="syntax error"),
            attempt_record(sql="SELECT n", executed_sql="SELECT n LIMIT 100", index=1),
        ],
        frame=frame_to_state(frame()),
        executed_sql="SELECT n LIMIT 100",
    )

    answer = answer_from_state("5823.", state)

    assert answer.sql == "SELECT n LIMIT 100"


def test_truncation_reaches_the_scorer():
    """A capped result compared against an uncapped reference is a false failure
    the report has to be able to explain."""
    state = state_with(frame=frame_to_state(frame(truncated=True)), executed_sql="q")

    assert answer_from_state("...", state).truncated is True


def test_the_trios_consulted_are_reported():
    """Eight eval cases turn on whether the corpus was used at all."""
    state = state_with(trio_ids=["trio-loyal", "trio-churn"])

    assert answer_from_state("...", state).trios == ("trio-loyal", "trio-churn")


def test_tool_calls_are_counted_for_the_cost_column():
    state = state_with(
        calls=3,
        events=[step_event(name, 0.0) for name in ("analyst", "run_sql", "run_sql")],
    )

    assert answer_from_state("...", state).calls == 3


def test_the_tools_are_reported_in_the_order_they_ran():
    """The suite scores the ordering — asked before queried — so the sequence
    is the part that has to survive, not the set."""
    state = state_with(
        events=[
            step_event(name, 0.0)
            for name in ("ask_for_definitions", "analyst", "run_sql")
        ]
    )

    assert answer_from_state("...", state).tools == (
        "ask_for_definitions",
        "analyst",
        "run_sql",
    )


def test_a_generated_report_reaches_the_eval():
    """The supervisor's answer is now a covering line, so a report case that
    scored `text` would score the covering line and pass on an empty report.

    `report_bodies` is supplied by the caller — state never carries a body —
    which is `ask_once`'s job in `seams.py`; here it stands in for that read.
    """
    state = state_with(
        reports_written=[{"report_id": "7f3a", "title": "Q1 Denim", "show": True}]
    )

    answer = answer_from_state(
        "Written and saved.", state, report_bodies=("## Summary\nDenim fell.",)
    )

    assert answer.report_ids == ("7f3a",)
    assert answer.report_bodies == ("## Summary\nDenim fell.",)
