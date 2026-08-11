"""Terminal presentation.

Rendered through a real `Console` rather than asserted on the arguments: a
`Panel` stringifies to `<rich.panel.Panel object at 0x...>`, so argument-level
assertions pass on output a user cannot read.
"""

import io

from langchain_core.messages import HumanMessage
from rich.console import Console

from retail_agent.agent.state import attempt_record, step_event
from retail_agent.cli.render import (
    render_answer,
    render_confirmation,
    render_error,
    render_metrics,
    render_reports,
    render_trace,
)
from retail_agent.obs.traces import trace_from_state


def recorder():
    return Console(record=True, width=100, file=io.StringIO())


def text(console) -> str:
    return console.export_text(clear=False)


def state_with(**overrides) -> dict:
    base = {
        "messages": [],
        "attempts": [],
        "events": [],
        "trio_ids": [],
        "assumed_terms": [],
        "preference_changes": [],
        "reports_written": [],
        "redactions": 0,
    }
    base.update(overrides)
    return base


def trace_for(question, answer, **overrides):
    state = state_with(messages=[HumanMessage(content=question)], **overrides)
    return trace_from_state(
        state, answer, user_id="dana", session_id="s1", turn_id="t1"
    )


def test_an_empty_answer_prints_nothing():
    """A failed turn already rendered its error; a blank panel under it is noise."""
    console = recorder()
    render_answer(console, "", state_with())

    assert text(console).strip() == ""


def test_the_footnote_admits_masking_and_retries():
    console = recorder()
    state = state_with(
        redactions=2,
        attempts=[
            attempt_record(sql="bad", error="boom"),
            attempt_record(sql="good", executed_sql="good", row_count=1, index=1),
        ],
    )

    render_answer(console, "Two customers.", state)

    printed = text(console)
    assert "2 personal-data values masked" in printed
    assert "2 query attempts" in printed


def test_a_clean_single_query_turn_has_no_footnote():
    """The footnote is a signal. Printed on every turn it stops being one."""
    console = recorder()
    state = state_with(attempts=[attempt_record(sql="q", executed_sql="q", row_count=1)])

    render_answer(console, "One.", state)

    assert "query attempts" not in text(console)


def test_the_footnote_can_be_turned_off():
    from retail_agent.store.preferences import Preferences

    console = recorder()
    state = state_with(
        redactions=1,
        attempts=[
            attempt_record(sql="a", error="x"),
            attempt_record(sql="b", executed_sql="b", row_count=1, index=1),
        ],
    )

    render_answer(
        console, "One.", state, prefs=Preferences(show_attempt_footnote=False)
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
    state = state_with(
        messages=[HumanMessage(content="how many?")],
        attempts=[
            attempt_record(sql="SELECT *", violations=["SELECT * is not allowed."]),
            attempt_record(
                sql="SELECT n FROM t",
                executed_sql="SELECT n FROM `ds.t`",
                row_count=1,
                bytes_billed=2_048,
                index=1,
            ),
        ],
        events=[step_event("run_sql", 0.0, "1 row(s)")],
    )

    render_trace(
        console,
        trace_from_state(state, "One.", user_id="dana", session_id="s1", turn_id="t1"),
    )

    printed = text(console)
    assert "SELECT * is not allowed." in printed
    assert "ds.t" in printed, "the query the warehouse actually saw"
    assert "2048" in printed
    assert "run_sql" in printed


def test_the_trace_shows_the_answer_it_stored():
    """Stored, truncated and read back, then never shown. `/trace <id>` on an
    older turn could say what ran but not what was said."""
    console = recorder()

    render_trace(console, trace_for("how many?", "Nine customers."))

    assert "Nine customers." in text(console)


def test_a_turn_that_called_no_tool_says_so_rather_than_printing_an_empty_table():
    """Easy to reach on a follow-up the model answers from the conversation.
    Headers with nothing under them read as a broken renderer."""
    console = recorder()

    render_trace(console, trace_for("and the month before?", "It was 11."))

    printed = text(console)
    assert "no tools" in printed
    assert "what happened" not in printed, "no empty table"


def test_the_trace_names_the_definitions_used_and_the_terms_assumed():
    console = recorder()
    state = state_with(
        messages=[HumanMessage(content="who is loyal?")],
        trio_ids=["trio-loyalty"],
        assumed_terms=["loyal"],
        preference_changes=[{"action": "answer_format", "note": "bullets"}],
        events=[step_event("analyst", 0.0, "1 trio(s)")],
    )

    render_trace(
        console,
        trace_from_state(state, "Nine.", user_id="dana", session_id="s1", turn_id="t1"),
    )

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
    return {"report_id": "7f3a", "title": title, "body": body, "show": show}


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
