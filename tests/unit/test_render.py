from rich.console import Console

from retail_agent.agent.state import SqlAttempt, fresh_scratch, new_turn_state
from retail_agent.cli.render import render_answer, render_error


def state_with(**overrides):
    state = new_turn_state(user_id="d", session_id="s", question="q")
    state.update(fresh_scratch(repair_budget=2))
    state.update(overrides)
    return state


def rendered(state) -> str:
    console = Console(record=True, width=100)
    render_answer(console, state)
    return console.export_text()


def test_answer_is_printed():
    assert "Revenue was $1.2M." in rendered(state_with(answer="Revenue was $1.2M."))


def test_nothing_is_printed_without_an_answer():
    assert rendered(state_with()).strip() == ""


def test_degraded_status_is_signalled():
    text = rendered(state_with(answer="I couldn't do that.", status="degraded"))
    assert "partial" in text.lower()


def test_redaction_count_is_surfaced():
    assert "12" in rendered(state_with(answer="Top customers listed.", redactions=12))


def test_retry_count_is_surfaced():
    state = state_with(
        answer="Done.",
        sql_attempts=[
            SqlAttempt(step_id="s", sql="a", violations=("x",)),
            SqlAttempt(step_id="s", sql="b", row_count=1),
        ],
    )
    assert "2 query attempts" in rendered(state)


def test_error_panel_includes_the_turn_id():
    console = Console(record=True, width=100)
    render_error(console, "It broke.", turn_id="abc123")
    text = console.export_text()

    assert "It broke." in text
    assert "abc123" in text


def _action(count):
    from retail_agent.agent.state import PendingAction

    return PendingAction(
        action_id="a1",
        report_ids=tuple(f"id{i}" for i in range(count)),
        titles=tuple(f"Report {i}" for i in range(count)),
        token="y" if count == 1 else f"DELETE {count}",
    )


def test_manifest_lists_the_titles_and_the_required_token():
    from retail_agent.cli.render import render_manifest

    console = Console(record=True, width=100)
    render_manifest(console, _action(2))
    output = console.export_text()

    assert "Report 0" in output
    assert "Report 1" in output
    assert "DELETE 2" in output


def test_manifest_shows_every_title_it_is_given():
    """Truncating would mean confirming a deletion you cannot see."""
    from retail_agent.cli.render import render_manifest

    console = Console(record=True, width=100)
    render_manifest(console, _action(30))
    output = console.export_text()

    assert "Report 0" in output
    assert "Report 29" in output


def _traced_state():
    from retail_agent.agent.state import SqlAttempt, TurnEvent

    return {
        "turn_id": "abc123def456",
        "intent": "analyze",
        "status": "ok",
        "redactions": 4,
        "events": [
            TurnEvent(node="start_turn", duration_ms=0),
            TurnEvent(node="route", duration_ms=310, detail="intent=analyze"),
            TurnEvent(node="plan", duration_ms=900, detail="2 step(s)"),
            TurnEvent(
                node="draft_sql",
                duration_ms=1200,
                detail="step_1: guard rejected — Column 'email' is personal data.",
            ),
            TurnEvent(node="execute", duration_ms=450, detail="step_1: 10 row(s)"),
        ],
        "sql_attempts": [
            SqlAttempt(
                step_id="step_1",
                sql="SELECT email AS c FROM users",
                violations=("Column 'email' is personal data.",),
            ),
            SqlAttempt(
                step_id="step_1",
                sql="SELECT id FROM users",
                executed_sql="SELECT id FROM `ds.users` LIMIT 500",
                row_count=10,
                bytes_billed=2048,
            ),
        ],
    }


def test_trace_shows_the_turn_id_and_every_node_in_order():
    from retail_agent.cli.render import render_trace

    console = Console(record=True, width=110)
    render_trace(console, _traced_state())
    output = console.export_text()

    assert "abc123def456" in output
    for node in ("start_turn", "route", "plan", "draft_sql", "execute"):
        assert node in output
    assert output.index("route") < output.index("plan") < output.index("execute")


def test_trace_shows_why_a_query_was_rejected():
    """The whole point: "3 query attempts" does not say what went wrong."""
    from retail_agent.cli.render import render_trace

    console = Console(record=True, width=110)
    render_trace(console, _traced_state())
    output = console.export_text()

    assert "guard rejected" in output
    assert "email" in output


def test_trace_shows_both_the_drafted_and_the_executed_sql():
    from retail_agent.cli.render import render_trace

    console = Console(record=True, width=110)
    render_trace(console, _traced_state())
    output = console.export_text()

    assert "SELECT email AS c FROM users" in output
    assert "`ds.users`" in output, "the query that actually ran"


def test_trace_on_a_turn_with_no_events_says_so():
    from retail_agent.cli.render import render_trace

    console = Console(record=True, width=110)
    render_trace(console, {})
    assert "no turn" in console.export_text().lower()


def test_the_attempt_count_excludes_the_diagnosis_marker():
    """A diagnosed turn drafts twice and runs twice. Recording the diagnosis as
    a failed attempt is what lets draft_sql reuse its repair prompt, but the
    user must not be told the agent tried three times."""
    from retail_agent.agent.state import SqlAttempt

    state = state_with(
        answer="Revenue was $10,650.80.",
        sql_attempts=[
            SqlAttempt(step_id="step_1", sql="SELECT 1", row_count=1),
            SqlAttempt(
                step_id="step_1", sql="SELECT 1", error="empty", is_diagnosis=True
            ),
            SqlAttempt(step_id="step_1", sql="SELECT 2", row_count=4),
        ],
    )

    assert "2 query attempts" in rendered(state)
    assert "3 query attempts" not in rendered(state)
