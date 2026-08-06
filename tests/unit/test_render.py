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
