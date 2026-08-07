"""Reducing a ReAct turn to something the eval can score.

The load-bearing test is the last one. Both arms are scored by the same
`run_case`, so if the two reductions disagree about what a result *was*, every
number in the comparison is measuring the reducers rather than the agents.
"""

from __future__ import annotations

import pandas as pd

from retail_agent.agent.state import AnalysisStep, MaskedFrame, SqlAttempt
from retail_agent.baseline.capture import ResultCapture
from retail_agent.baseline.seams import answer_from_react
from retail_agent.evals.harness import answer_from_state

SQL = "SELECT SUM(sale_price) AS total FROM order_items"


def frame(rows: pd.DataFrame, *, row_count: int | None = None) -> MaskedFrame:
    return MaskedFrame.from_dataframe(
        rows,
        row_count=row_count if row_count is not None else len(rows),
        redactions=0,
        truncated=row_count is not None and row_count > len(rows),
    )


def test_a_turn_that_ran_nothing_scores_as_no_rows():
    answer = answer_from_react("I could not answer that.", ResultCapture())

    assert answer.rows == []
    assert answer.sql == ""
    assert answer.text == "I could not answer that."


def test_the_number_comes_from_the_frame_not_the_narrative():
    """The model's prose may round, hedge or reword. The frame is what the
    warehouse returned."""
    capture = ResultCapture()
    capture.record_query(SQL, frame(pd.DataFrame({"total": [1234.5]})))

    answer = answer_from_react("Revenue was about 1.2k.", capture)

    assert answer.rows == [[1234.5]]
    assert answer.columns == ("total",)


def test_a_capped_result_reports_the_true_size():
    """`truncated` is what stops `run_case` scoring a sample as the answer."""
    capture = ResultCapture()
    capture.record_query(SQL, frame(pd.DataFrame({"id": [1, 2]}), row_count=5823))

    answer = answer_from_react("...", capture)

    assert answer.truncated is True
    assert answer.row_count == 5823


def test_both_arms_reduce_the_same_result_to_the_same_answer():
    """The equivalence that makes the comparison trustworthy.

    One `MaskedFrame`, one executed query, one consulted trio — expressed once
    as graph state and once as a ReAct capture. Every field `run_case` reads
    must come out identical.
    """
    result = frame(pd.DataFrame({"total": [1234.5]}))

    state = {
        "answer": "Revenue was 1,234.50.",
        "intent": "analyze",
        "trio_ids": ["loyal_v1"],
        "plan": [AnalysisStep(id="step_1", question="revenue?", sql=SQL)],
        "frames": {"step_1": result},
        "sql_attempts": [SqlAttempt(step_id="step_1", sql=SQL, executed_sql=SQL)],
    }

    capture = ResultCapture()
    capture.record_definitions(["loyal_v1"])
    capture.record_query(SQL, result)

    from_graph = answer_from_state(state)
    from_react = answer_from_react("Revenue was 1,234.50.", capture)

    assert from_react.rows == from_graph.rows
    assert from_react.columns == from_graph.columns
    assert from_react.sql == from_graph.sql
    assert from_react.row_count == from_graph.row_count
    assert from_react.truncated == from_graph.truncated
    assert from_react.trios == from_graph.trios
    assert from_react.text == from_graph.text
