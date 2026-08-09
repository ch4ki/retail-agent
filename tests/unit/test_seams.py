"""Reducing a finished turn to something an eval can score.

This is a pure function on purpose. It is the one part of the eval wiring that
decides what "the agent's answer" means, and a rule that could only be exercised
by a live run against BigQuery would never be checked at all.
"""

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.seams import answer_from_capture
from retail_agent.safety.frame import MaskedFrame


def frame(**kwargs) -> MaskedFrame:
    defaults = dict(columns=("n",), rows=((5_823,),), row_count=5_823, redactions=0)
    return MaskedFrame(**{**defaults, **kwargs})


def test_the_number_comes_from_the_frame_and_not_from_the_prose():
    """Parsing prose would measure how the model phrased itself, which varies
    legitimately between model versions, rather than what the query returned."""
    capture = TurnCapture()
    capture.record_attempt("SELECT COUNT(*)", executed_sql="SELECT COUNT(*)", frame=frame())

    answer = answer_from_capture("About six thousand, give or take.", capture)

    assert answer.rows == [[5_823]]
    assert answer.row_count == 5_823
    assert answer.text == "About six thousand, give or take."


def test_a_turn_that_never_queried_scores_as_no_rows_rather_than_raising():
    """A refused or chat turn is a legitimate outcome, not a harness error."""
    answer = answer_from_capture("I can only help with the retail data.", TurnCapture())

    assert answer.rows == []
    assert answer.columns == ()
    assert answer.sql == ""
    assert answer.intent == "chat"


def test_the_scored_query_is_the_one_that_ran():
    """A repaired turn has several attempts and only the last produced a number."""
    capture = TurnCapture()
    capture.record_attempt("SELECT bad", error="syntax error")
    capture.record_attempt("SELECT n", executed_sql="SELECT n LIMIT 100", frame=frame())

    answer = answer_from_capture("5823.", capture)

    assert answer.sql == "SELECT n LIMIT 100"


def test_truncation_reaches_the_scorer():
    """A capped result compared against an uncapped reference is a false failure
    the report has to be able to explain."""
    capture = TurnCapture()
    capture.record_attempt("q", executed_sql="q", frame=frame(truncated=True))

    assert answer_from_capture("...", capture).truncated is True


def test_the_trios_consulted_are_reported():
    """Eight eval cases turn on whether the corpus was used at all."""
    capture = TurnCapture()
    capture.record_definitions(["trio-loyal", "trio-churn"])

    assert answer_from_capture("...", capture).trios == ("trio-loyal", "trio-churn")


def test_tool_calls_are_counted_for_the_cost_column():
    capture = TurnCapture()
    for name in ("analyst", "run_sql", "run_sql"):
        with capture.step(name):
            pass

    assert answer_from_capture("...", capture).calls == 3


def test_the_tools_are_reported_in_the_order_they_ran():
    """The suite scores the ordering — asked before queried — so the sequence
    is the part that has to survive, not the set."""
    capture = TurnCapture()
    for name in ("ask_for_definitions", "analyst", "run_sql"):
        with capture.step(name):
            pass

    assert answer_from_capture("...", capture).tools == (
        "ask_for_definitions",
        "analyst",
        "run_sql",
    )
