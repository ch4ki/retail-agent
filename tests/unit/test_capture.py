"""What a turn remembers about itself."""

import pandas as pd

from retail_agent.agent.capture import TurnCapture
from retail_agent.safety.frame import MaskedFrame


def frame(**values) -> MaskedFrame:
    return MaskedFrame.from_dataframe(
        pd.DataFrame(values), row_count=len(next(iter(values.values()))), redactions=0
    )


def test_a_failed_attempt_is_recorded_but_does_not_become_the_answer():
    """`first_pass_validity` divides by attempts, so failures have to be there —
    and `frame` has to stay empty, or a rejected query would be scored."""
    capture = TurnCapture()
    capture.record_attempt("SELECT *", violations=("SELECT * is not allowed.",))

    assert len(capture.attempts) == 1
    assert capture.frame is None
    assert capture.executed_sql == ""


def test_the_last_successful_query_wins():
    capture = TurnCapture()
    capture.record_attempt("bad", error="boom")
    capture.record_attempt("good", executed_sql="good limited", frame=frame(n=[1]))

    assert capture.executed_sql == "good limited"
    assert capture.frame.rows == ((1,),)


def test_redactions_accumulate_across_queries():
    """The footnote counts the turn, not the last query in it."""
    capture = TurnCapture()
    for _ in range(2):
        capture.record_attempt(
            "q",
            executed_sql="q",
            frame=MaskedFrame(columns=("a",), rows=((1,),), row_count=1, redactions=3),
        )

    assert capture.redactions == 6


def test_definitions_are_reported_once_in_call_order():
    capture = TurnCapture()
    capture.record_definitions(["trio-b", "trio-a"])
    capture.record_definitions(["trio-b"])

    assert capture.trio_ids == ["trio-b", "trio-a"]


def test_intent_is_derived_from_what_actually_ran():
    """The graph asked a model this before doing any of it. Which tools ran is
    the same answer, arrived at afterwards and for free."""
    analysis = TurnCapture()
    with analysis.step("run_sql"):
        pass
    assert analysis.intent == "analyze"

    library = TurnCapture()
    with library.step("list_reports"):
        pass
    assert library.intent == "report_op"

    assert TurnCapture().intent == "chat"


def test_a_step_is_timed_even_when_it_raises():
    """A rejected query is exactly the step `/trace` is opened to look at."""
    capture = TurnCapture()
    try:
        with capture.step("run_sql") as step:
            step.detail = "guard rejected"
            raise ValueError("no")
    except ValueError:
        pass

    assert [name for name, _, _ in capture.events] == ["run_sql"]
    assert capture.events[0][2] == "guard rejected"


def test_the_trace_carries_the_attempts_and_no_rows():
    capture = TurnCapture(user_id="exec", session_id="s1", question="how many?")
    capture.record_attempt(
        "SELECT 1", executed_sql="SELECT 1", frame=frame(email=["a@b.com"]),
        bytes_billed=2_000,
    )

    trace = capture.to_trace("There is one.")

    assert trace.owner_id == "exec"
    assert trace.bytes_billed == 2_000
    assert trace.attempts[0]["executed_sql"] == "SELECT 1"
    assert "a@b.com" not in str(trace)


def test_a_long_answer_is_truncated_in_the_trace():
    """A trace is for debugging, not a second copy of every report."""
    trace = TurnCapture().to_trace("x" * 10_000)

    assert len(trace.answer) == 4_000
