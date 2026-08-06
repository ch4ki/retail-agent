"""Turning a finished turn into something scorable.

The extraction is pure and is tested here; the wiring that builds deps and talks
to BigQuery is exercised by `-m live`.
"""

from __future__ import annotations

from retail_agent.agent.state import AnalysisStep, MaskedFrame, SqlAttempt
from retail_agent.evals.harness import answer_from_state


def step(step_id: str) -> AnalysisStep:
    return AnalysisStep(id=step_id, question="q")


def frame(rows, columns=("n",)) -> MaskedFrame:
    return MaskedFrame(
        columns=columns, rows=tuple(tuple(r) for r in rows), row_count=len(rows), redactions=0
    )


def test_the_answer_comes_from_the_last_step_of_the_plan():
    """A multi-step plan leaves several frames. The final one is the answer;
    an earlier one is working."""
    state = {
        "answer": "5,746 customers.",
        "frames": {"s1": frame([[100]]), "s2": frame([[5746]])},
        "plan": [step("s1"), step("s2")],
    }

    assert answer_from_state(state).rows == [[5746]]


def test_a_turn_with_no_frames_yields_no_rows():
    """Chat and schema questions answer without SQL, and a degraded turn has
    nothing. Both must produce an empty answer rather than raise."""
    state = {"answer": "I could not answer that.", "frames": {}, "plan": []}

    assert answer_from_state(state).rows == []


def test_the_sql_recorded_is_the_one_that_ran():
    """Not the first draft. A repaired turn has several attempts and only the
    last one produced the number being scored."""
    state = {
        "answer": "",
        "frames": {"s1": frame([[1]])},
        "plan": [step("s1")],
        "sql_attempts": [
            SqlAttempt(step_id="s1", sql="SELECT bad", error="boom"),
            SqlAttempt(step_id="s1", sql="SELECT good", executed_sql="SELECT good", row_count=1),
        ],
    }

    assert answer_from_state(state).sql == "SELECT good"


def test_trios_used_are_carried_through():
    """So a failure can be read as "it had the definition and still got it
    wrong" rather than "it never retrieved one"."""
    state = {
        "answer": "",
        "frames": {},
        "plan": [],
        "trio_ids": ["loyal-customers"],
    }

    assert answer_from_state(state).trios == ("loyal-customers",)


def test_the_narrative_answer_is_kept_for_the_pii_scan():
    state = {"answer": "Contact bob@example.com", "frames": {}, "plan": []}

    assert "bob@example.com" in answer_from_state(state).text


def test_frames_are_used_even_when_the_plan_is_missing():
    """Defensive: a degraded turn can leave frames without a readable plan, and
    losing the number then would report ERROR for a turn that did answer."""
    state = {"answer": "", "frames": {"only": frame([[42]])}, "plan": []}

    assert answer_from_state(state).rows == [[42]]
