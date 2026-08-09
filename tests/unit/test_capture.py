"""What a turn files about itself on the way past."""

from retail_agent.agent.capture import TurnCapture


def test_a_written_report_is_kept_with_its_exact_body():
    """The CLI prints this copy rather than anything the model produced, so a
    byte that changes here is a byte that differs from what was stored."""
    capture = TurnCapture(user_id="exec")
    body = "## Summary\nDenim fell in Q1.\n"

    capture.record_report("7f3a", "Q1 Denim", body, show=True)

    assert len(capture.reports_written) == 1
    written = capture.reports_written[0]
    assert written.report_id == "7f3a"
    assert written.title == "Q1 Denim"
    assert written.body == body
    assert written.show is True


def test_writing_a_report_is_a_report_operation():
    """`save_report` used to be what marked these turns; it no longer exists."""
    capture = TurnCapture(user_id="exec")
    with capture.step("report_writer"):
        pass

    assert capture.intent == "report_op"


def test_context_tokens_default_to_zero():
    """A failed turn never reaches the recorder, so the field must read as
    'not measured' rather than as a missing attribute."""
    assert TurnCapture().context_tokens == 0
