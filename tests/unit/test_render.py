from rich.console import Console

from retail_agent.agent.state import SqlAttempt, new_turn_state
from retail_agent.cli.render import render_answer, render_error


def state_with(**overrides):
    state = new_turn_state(user_id="d", session_id="s", question="q", repair_budget=2)
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
