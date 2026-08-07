"""What the ReAct arm remembers of its own queries.

A ReAct agent has only a message list, and the rows the model saw were formatted
into a `ToolMessage` string. Scoring that string would measure formatting, so
`run_sql` records the real frame here instead and the eval reads it back.
"""

from __future__ import annotations

import pandas as pd

from retail_agent.baseline.capture import ResultCapture
from retail_agent.agent.state import MaskedFrame


def frame(value: int) -> MaskedFrame:
    return MaskedFrame.from_dataframe(
        pd.DataFrame({"total": [value]}), row_count=1, redactions=0
    )


def test_an_untouched_capture_has_no_result():
    capture = ResultCapture()

    assert capture.frame is None
    assert capture.executed_sql == ""


def test_the_most_recent_query_is_the_one_scored():
    """A repaired turn runs several queries and only the last produced the
    number being scored — the same rule `_executed_sql` applies to the graph."""
    capture = ResultCapture()

    capture.record_query("SELECT 1", frame(1))
    capture.record_query("SELECT 2", frame(2))

    assert capture.executed_sql == "SELECT 2"
    assert capture.frame.rows == ((2,),)


def test_definition_lookups_are_recorded_separately_from_queries():
    """`AgentAnswer.trios` reports which trios were consulted. On the graph arm
    that is `trio_ids` in state; here it is whatever the model chose to look up."""
    capture = ResultCapture()

    capture.record_definitions(["loyal_customer", "churn_window"])
    capture.record_definitions(["loyal_customer"])

    assert capture.trio_ids == ("loyal_customer", "churn_window")


def test_every_tool_invocation_is_counted():
    """`calls` is the ReAct arm's answer to the graph's `events` — what the
    comparison uses to say an answer cost three times as many round trips."""
    capture = ResultCapture()

    capture.record_query("SELECT 1", frame(1))
    capture.record_definitions(["loyal_customer"])
    capture.record_query("SELECT 2", frame(2))

    assert capture.calls == 3
