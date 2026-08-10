"""Terminal presentation.

Rendered through a real `Console` rather than asserted on the arguments: a
`Panel` stringifies to `<rich.panel.Panel object at 0x...>`, so argument-level
assertions pass on output a user cannot read.
"""

import io

from rich.console import Console

from retail_agent.agent.capture import TurnCapture, WrittenReport
from retail_agent.cli.render import (
    render_answer,
    render_confirmation,
    render_error,
    render_metrics,
    render_reports,
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
            "context_tokens_max": 0,
            "context_tokens_p50": 0,
        },
    )

    printed = text(console)
    assert "last 7 turn(s)" in printed
    assert "run_sql" in printed


def written(title="Q1 Denim", body="## Summary\nDenim fell 11.8% in Q1.", show=True):
    return WrittenReport(report_id="7f3a", title=title, body=body, show=show)


def test_the_report_is_printed_whole_with_where_it_was_saved():
    """The executive asked for a report. Printing a filing receipt instead is
    the failure this exists to fix."""
    console = recorder()

    render_reports(console, [written()])

    printed = text(console)
    assert "Denim fell 11.8% in Q1." in printed
    assert "7f3a" in printed
    assert "Q1 Denim" in printed


def test_a_hidden_draft_is_not_printed():
    """`show_to_executive=False` is the model saying the executive did not ask
    to read this one. The CLI obeys rather than second-guessing it."""
    console = recorder()

    render_reports(console, [written(body="Interim numbers.", show=False)])

    assert "Interim numbers." not in text(console)


def test_a_turn_that_wrote_nothing_prints_nothing():
    """Most turns write no report. A blank line under every answer is noise."""
    console = recorder()

    render_reports(console, [])

    assert text(console).strip() == ""


def test_the_trace_names_the_reports_the_turn_produced():
    """The report body no longer passes through the answer, so without this a
    turn that wrote one leaves no link to it."""
    from retail_agent.obs.traces import TraceRecord

    console = recorder()
    render_trace(
        console,
        TraceRecord(
            turn_id="t1",
            session_id="s1",
            owner_id="exec",
            question="write me a report",
            intent="report_op",
            status="ok",
            answer="Written and saved.",
            report_ids=["7f3a"],
            context_tokens=4_200,
        ),
    )

    printed = text(console)
    assert "7f3a" in printed
    assert "4200" in printed.replace(",", "")


def test_metrics_report_how_large_conversations_actually_get():
    """A maximum as well as a median: the threshold has to clear the worst
    session, not the typical one."""
    console = recorder()
    render_metrics(
        console,
        {
            "turns": 3,
            "first_pass_validity": 1.0,
            "self_correction_rate": 0.0,
            "redactions": 0,
            "bytes_billed": 0,
            "node_p50_ms": {},
            "context_tokens_max": 9_000,
            "context_tokens_p50": 4_000,
        },
    )

    printed = text(console).replace(",", "")
    assert "9000" in printed
    assert "4000" in printed


# --- /prefs ---


def test_prefs_shows_the_notes_the_agent_saved():
    """The gap this closes, reported from a live session.

    Told "from now on keep it brief", the agent saves a note and the CLI
    announces it — ending with "/prefs to change it". `/prefs` then printed one
    typed setting and no notes, so the thing the user had just changed was the
    one thing the screen it pointed at did not show. A preference you cannot
    see is one you cannot withdraw.
    """
    from retail_agent.cli.render import render_preferences
    from retail_agent.store.preferences import Preferences

    console = recorder()

    render_preferences(
        console, Preferences(), notes=["Keep answers brief and give just the numbers."]
    )

    assert "Keep answers brief" in text(console)
    assert "forget" in text(console).lower(), "and how to drop one"


def test_prefs_with_no_notes_says_so_rather_than_showing_an_empty_table():
    from retail_agent.cli.render import render_preferences
    from retail_agent.store.preferences import Preferences

    console = recorder()

    render_preferences(console, Preferences(), notes=[])

    assert "show_attempt_footnote" in text(console), "the typed settings still render"
