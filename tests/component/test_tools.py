"""The tools that reach the warehouse.

`run_sql` is the whole PII boundary. These are the tests that keep it one.
"""

import inspect

import pandas as pd
import pytest

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import TurnContext
from retail_agent.agent.tools import EMPTY_HINT, GuardRejection, build_analyst_tools
from retail_agent.datasources.base import QuerySyntaxError

from .conftest import FakeSource


def _runtime():
    """A `ToolRuntime` good enough to call a tool's raw `.func` directly.

    Calling `.func` bypasses `BaseTool.run` and the runtime injection that
    goes with it, so the test has to build one itself.
    """
    from langchain.tools import ToolRuntime

    return ToolRuntime(
        state=None,
        context=TurnContext(user_id="exec", session_id="s1"),
        config={},
        stream_writer=None,
        tool_call_id="test",
        store=None,
    )


def tools_for(deps):
    import functools

    capture = TurnCapture(user_id="exec", session_id="s1", question="q")
    by_name = {
        t.name: functools.partial(t.func, runtime=_runtime())
        for t in build_analyst_tools(deps, capture)
    }
    return by_name, capture


def test_a_guard_violation_never_reaches_the_warehouse(make_deps, source):
    """The order matters more than the message: a rejected query must not run.

    Asserting on `source.executed` rather than on the exception is deliberate —
    a guard that raised *after* executing would still raise.
    """
    deps = make_deps(src=source)
    tools, capture = tools_for(deps)

    with pytest.raises(GuardRejection):
        tools["run_sql"]("DROP TABLE users")

    assert source.executed == []
    assert capture.attempts[-1]["violations"]


def test_restricted_columns_are_dropped_before_the_model_sees_them(make_deps):
    """`email` is in the policy, so the rendered table cannot contain it."""
    source = FakeSource(
        frames={
            "default": pd.DataFrame(
                {"id": [1], "email": ["a@b.com"], "spend": [100]}
            )
        }
    )
    deps = make_deps(src=source)
    tools, capture = tools_for(deps)

    rendered = tools["run_sql"]("SELECT id, spend FROM users")

    assert "a@b.com" not in rendered
    assert capture.frame is not None


def test_only_run_sql_reads_the_warehouse(make_deps):
    """The property the graph got from edge order, kept by construction.

    If a tool is added that queries BigQuery outside `run_sql`, masking is no
    longer guaranteed for anything it returns — and nothing else in the suite
    would notice.
    """
    deps = make_deps()
    from retail_agent.agent.memory import build_memory_tools
    from retail_agent.agent.reports import build_report_tools
    from retail_agent.agent.schema import build_schema_tool

    capture = TurnCapture()
    others = [
        *build_report_tools(deps, capture),
        *build_memory_tools(deps, capture),
        *build_schema_tool(deps, capture),
        build_analyst_tools(deps, capture)[1],  # lookup_definitions
    ]

    for t in others:
        body = inspect.getsource(t.func)
        assert "source.execute" not in body, f"{t.name} queries the warehouse"


def test_a_failed_query_is_recorded_and_re_raised(make_deps):
    """`ToolErrorMiddleware` needs the exception; `/trace` needs the attempt."""
    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})}, failing={"bad"})
    deps = make_deps(src=source)
    tools, capture = tools_for(deps)

    with pytest.raises(QuerySyntaxError):
        tools["run_sql"]("SELECT bad FROM users")

    assert capture.attempts[-1]["error"]
    assert capture.frame is None


def test_the_last_successful_query_is_the_one_kept(make_deps):
    """A turn that fails, repairs and succeeds is scored on the query that ran."""
    source = FakeSource(
        frames={"default": pd.DataFrame({"n": [7]})}, failing={"broken"}
    )
    deps = make_deps(src=source)
    tools, capture = tools_for(deps)

    with pytest.raises(QuerySyntaxError):
        tools["run_sql"]("SELECT broken FROM orders")
    tools["run_sql"]("SELECT n FROM orders")

    assert len(capture.attempts) == 2
    assert "broken" not in capture.executed_sql
    assert capture.frame.rows == ((7,),)


def test_an_empty_result_carries_the_hint_the_graph_spent_a_call_on(make_deps):
    """Zero rows raises nothing, so no retry middleware reacts to it."""
    source = FakeSource(frames={"default": pd.DataFrame()}, empty_for={"Levis"})
    deps = make_deps(src=source)
    tools, _ = tools_for(deps)

    rendered = tools["run_sql"]("SELECT id FROM products WHERE brand = 'Levis'")

    assert EMPTY_HINT.strip() in rendered


def test_an_all_null_aggregate_counts_as_empty(make_deps):
    """`SUM(x)` over no rows returns one row of NULL, not zero rows.

    Checking `row_count == 0` alone misses the case the hint exists for.
    """
    source = FakeSource(
        frames={"default": pd.DataFrame()}, null_aggregate_for={"Levis"}
    )
    deps = make_deps(src=source)
    tools, _ = tools_for(deps)

    rendered = tools["run_sql"](
        "SELECT SUM(sale_price) AS total_revenue FROM order_items WHERE brand = 'Levis'"
    )

    assert EMPTY_HINT.strip() in rendered


def test_a_capped_result_says_so(make_deps):
    """A sample read as a complete answer is how "20 loyal customers" was
    reported against a true 5,823."""
    source = FakeSource(
        frames={"default": pd.DataFrame({"id": [1, 2]})}, total_rows=5_823
    )
    deps = make_deps(src=source)
    tools, _ = tools_for(deps)

    rendered = tools["run_sql"]("SELECT id FROM users")

    assert "SAMPLE" in rendered
    assert "5823" in rendered or "5,823" in rendered


def test_lookup_definitions_says_so_when_nothing_covers_the_term(make_deps):
    """An empty string reads to a model as a definition of nothing."""
    deps = make_deps()
    tools, _ = tools_for(deps)

    assert "Decide for yourself" in tools["lookup_definitions"]("who is loyal?")
