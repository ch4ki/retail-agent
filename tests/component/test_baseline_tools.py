"""The two tools the ReAct arm gets, against a fake warehouse.

The point of the comparison is that both arms run the same guard and the same
masking. These tests hold that: a query the graph would reject must be rejected
here too, and by the same code rather than by a second implementation of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from retail_agent.baseline.capture import ResultCapture
from retail_agent.baseline.tools import GuardRejection, build_tools
from retail_agent.knowledge.trios import Trio

from .conftest import FakeSource


@pytest.fixture
def capture():
    return ResultCapture()


@pytest.fixture
def tools(make_deps, capture, source):
    deps = make_deps([], src=source)
    return {t.__name__: t for t in build_tools(deps, capture)}


def test_a_query_the_guard_rejects_never_reaches_the_warehouse(tools, source, capture):
    """The guard is a boundary, not advice. A rejected query must not execute,
    and must not leave a frame behind for the eval to score."""
    with pytest.raises(GuardRejection):
        tools["run_sql"]("DROP TABLE users")

    assert source.executed == []
    assert capture.frame is None


def test_a_successful_query_records_the_rows_it_returned(tools, capture):
    tools["run_sql"]("SELECT id, spend FROM order_items")

    assert capture.frame is not None
    assert capture.frame.row_count == 2
    assert capture.executed_sql != ""


def test_pii_never_reaches_the_model_or_the_capture(tools, capture):
    """`mask_dataframe` runs at the data boundary on this arm too. The fake
    source serves an `email` column, which the default policy does not allow."""
    rendered = tools["run_sql"]("SELECT id, email, spend FROM users")

    assert "a@b.com" not in rendered
    assert all("a@b.com" not in str(cell) for row in capture.frame.rows for cell in row)


def test_the_frame_scored_is_the_one_from_the_last_successful_query(make_deps, capture):
    """A repaired turn runs a broken query and then a working one. Only the
    second produced the number, which is the rule `_executed_sql` applies to
    the graph arm."""
    source = FakeSource(
        frames={"default": pd.DataFrame({"total": [42]})}, failing={"broken"}
    )
    tools = {t.__name__: t for t in build_tools(make_deps([], src=source), capture)}

    with pytest.raises(Exception):
        tools["run_sql"]("SELECT broken FROM order_items")
    tools["run_sql"]("SELECT total FROM order_items")

    assert capture.frame.rows == ((42,),)
    assert "broken" not in capture.executed_sql


def test_looking_up_definitions_returns_them_and_records_the_trios(make_deps, capture):
    trio = Trio(
        id="loyal_v1",
        question="who are our loyal customers?",
        sql="SELECT 1",
        report="...",
        metric_definitions={"loyal": "three or more orders in the trailing year"},
    )
    deps = make_deps([])
    tools = {
        t.__name__: t
        for t in build_tools(_with_trios(deps, [trio]), capture)
    }

    rendered = tools["lookup_definitions"]("who are our loyal customers?")

    assert "three or more orders" in rendered
    assert capture.trio_ids == ("loyal_v1",)


def test_a_question_no_trio_covers_says_so_rather_than_inventing_one(make_deps, capture):
    """An empty corpus is a valid state. The tool has to report nothing found,
    because a blank string reads to the model as a definition of nothing."""
    tools = {
        t.__name__: t for t in build_tools(_with_trios(make_deps([]), []), capture)
    }

    rendered = tools["lookup_definitions"]("what was revenue in May?")

    assert rendered.strip() != ""
    assert capture.trio_ids == ()


def _with_trios(deps, trios):
    from dataclasses import replace

    return replace(deps, trios=trios)
