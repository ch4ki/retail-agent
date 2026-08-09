"""Terminal presentation.

Rendered through a real `Console` rather than asserted on the arguments: a
`Panel` stringifies to `<rich.panel.Panel object at 0x...>`, so argument-level
assertions pass on output a user cannot read.
"""

import io

from rich.console import Console

from retail_agent.agent.capture import TurnCapture
from retail_agent.cli.render import (
    render_answer,
    render_confirmation,
    render_error,
    render_metrics,
    render_trace,
)
from retail_agent.safety.frame import MaskedFrame


def recorder():
    return Console(record=True, width=100, file=io.StringIO())


def text(console) -> str:
    return console.export_text(clear=False)


def frame(redactions=0, row_count=1):
    return MaskedFrame(
        columns=("n",), rows=((1,),), row_count=row_count, redactions=redactions
    )


def test_an_empty_answer_prints_nothing():
    """A failed turn already rendered its error; a blank panel under it is noise."""
    console = recorder()
    render_answer(console, "", TurnCapture())

    assert text(console).strip() == ""


def test_the_footnote_admits_masking_and_retries():
    console = recorder()
    capture = TurnCapture()
    capture.record_attempt("bad", error="boom")
    capture.record_attempt("good", executed_sql="good", frame=frame(redactions=2))

    render_answer(console, "Two customers.", capture)

    printed = text(console)
    assert "2 personal-data values masked" in printed
    assert "2 query attempts" in printed


def test_a_clean_single_query_turn_has_no_footnote():
    """The footnote is a signal. Printed on every turn it stops being one."""
    console = recorder()
    capture = TurnCapture()
    capture.record_attempt("q", executed_sql="q", frame=frame())

    render_answer(console, "One.", capture)

    assert "query attempts" not in text(console)


def test_the_footnote_can_be_turned_off():
    from retail_agent.store.preferences import Preferences

    console = recorder()
    capture = TurnCapture()
    capture.record_attempt("a", error="x")
    capture.record_attempt("b", executed_sql="b", frame=frame(redactions=1))

    render_answer(
        console, "One.", capture, prefs=Preferences(show_attempt_footnote=False)
    )

    assert "masked" not in text(console)


def test_the_confirmation_shows_the_manifest_verbatim():
    console = recorder()
    render_confirmation(console, "This will delete 2 report(s):\n  - Acme Q1\n  - Beta Q1")

    printed = text(console)
    assert "Acme Q1" in printed and "Beta Q1" in printed


def test_an_error_carries_the_turn_id():
    """A complaint becomes a single lookup rather than an investigation."""
    console = recorder()
    render_error(console, "The model timed out.", turn_id="abc123")

    assert "abc123" in text(console)


def test_trace_before_any_turn_explains_itself():
    console = recorder()
    render_trace(console, None)

    assert "ask a question first" in text(console)


def test_the_trace_shows_every_attempt_and_the_query_that_ran():
    console = recorder()
    capture = TurnCapture(user_id="dana", session_id="s1", question="how many?")
    capture.record_attempt("SELECT *", violations=("SELECT * is not allowed.",))
    capture.record_attempt(
        "SELECT n FROM t",
        executed_sql="SELECT n FROM `ds.t`",
        frame=frame(),
        bytes_billed=2_048,
    )
    with capture.step("run_sql") as step:
        step.detail = "1 row(s)"

    render_trace(console, capture.to_trace("One."))

    printed = text(console)
    assert "SELECT * is not allowed." in printed
    assert "ds.t" in printed, "the query the warehouse actually saw"
    assert "2048" in printed
    assert "run_sql" in printed


def test_the_trace_shows_the_answer_it_stored():
    """Stored, truncated and read back, then never shown. `/trace <id>` on an
    older turn could say what ran but not what was said."""
    console = recorder()
    capture = TurnCapture(user_id="dana", question="how many?")

    render_trace(console, capture.to_trace("Nine customers."))

    assert "Nine customers." in text(console)


def test_a_turn_that_called_no_tool_says_so_rather_than_printing_an_empty_table():
    """Easy to reach on a follow-up the model answers from the conversation.
    Headers with nothing under them read as a broken renderer."""
    console = recorder()
    capture = TurnCapture(user_id="dana", question="and the month before?")

    render_trace(console, capture.to_trace("It was 11."))

    printed = text(console)
    assert "no tools" in printed
    assert "what happened" not in printed, "no empty table"


def test_the_trace_names_the_definitions_used_and_the_terms_assumed():
    console = recorder()
    capture = TurnCapture(user_id="dana", question="who is loyal?")
    capture.record_definitions(["trio-loyalty"])
    capture.record_assumptions(["loyal"])
    capture.preference_changes.append(("answer_format", "bullets"))
    with capture.step("analyst") as step:
        step.detail = "1 trio(s)"

    render_trace(console, capture.to_trace("Nine."))

    printed = text(console)
    assert "trio-loyalty" in printed
    assert "loyal" in printed
    assert "answer_format" in printed and "bullets" in printed


def test_metrics_with_no_turns_says_so_rather_than_dividing_by_zero():
    console = recorder()
    render_metrics(console, {"turns": 0})

    assert "No turns recorded" in text(console)


def test_metrics_name_the_window_they_are_over():
    """"50% self-correction" over two turns is not the claim it looks like."""
    console = recorder()
    render_metrics(
        console,
        {
            "turns": 7,
            "first_pass_validity": 0.5,
            "self_correction_rate": 1.0,
            "redactions": 3,
            "bytes_billed": 1_234,
            "node_p50_ms": {"run_sql": 120},
        },
    )

    printed = text(console)
    assert "last 7 turn(s)" in printed
    assert "run_sql" in printed
