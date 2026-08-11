"""The tools that reach the warehouse.

`run_sql` is the whole PII boundary. These are the tests that keep it one.
"""

import inspect

import pandas as pd
import pytest

from retail_agent.agent.deps import TurnContext
from retail_agent.agent.tools import EMPTY_HINT, GuardRejection, build_analyst_tools
from retail_agent.datasources.base import QuerySyntaxError

from .conftest import FakeSource


def _runtime():
    """A `ToolRuntime` good enough to call a tool's raw `.func` directly.

    Calling `.func` bypasses `BaseTool.run` and the runtime injection that
    goes with it, so the test has to build one itself. `state` is a dict
    rather than `None` because `run_sql` now reads `runtime.state.get(
    "attempts", [])` to number its own `step_id` — a bare `.func` call never
    threads the graph's reducers, so this is always a fresh, empty turn as
    far as the tool can tell.
    """
    from langchain.tools import ToolRuntime

    return ToolRuntime(
        state={},
        context=TurnContext(user_id="exec", session_id="s1"),
        config={},
        stream_writer=None,
        tool_call_id="test",
        store=None,
    )


def tools_for(deps):
    import functools

    return {
        t.name: functools.partial(t.func, runtime=_runtime())
        for t in build_analyst_tools(deps)
    }


def _content(command):
    """The text a model would see, from a successful tool's `Command`.

    `run_sql` and `lookup_definitions` now return `Command` rather than a
    string; the rendered text lives in the one `ToolMessage` each puts on
    `update["messages"]`.
    """
    return command.update["messages"][0].content


def _sql_failure(deps, sql, state=None):
    """Drives a failing `run_sql` call through `_SqlFailureRecorder`, the
    middleware that now does what `TurnCapture.record_attempt` used to do
    inline, before re-raising.

    `run_sql` itself still just raises on this path — calling `.func`
    directly, as every other test in this file does, proves that half.
    Recording only happens one layer up, in the `wrap_tool_call` middleware
    that sits where `analyst_middleware` puts `_SqlFailureRecorder`, so
    exercising the recording means building the `ToolCallRequest` that
    layer receives and calling it the way `ToolNode` would.
    """
    from langchain.agents.middleware.types import ToolCallRequest

    from retail_agent.agent.middleware import _SqlFailureRecorder

    tool = build_analyst_tools(deps)[0]  # run_sql
    request = ToolCallRequest(
        tool_call={"name": "run_sql", "args": {"sql": sql}, "id": "test"},
        tool=tool,
        state=state or {},
        runtime=_runtime(),
    )

    def handler(req):
        return tool.func(req.tool_call["args"]["sql"], runtime=req.runtime)

    return _SqlFailureRecorder().wrap_tool_call(request, handler)


def test_a_guard_violation_never_reaches_the_warehouse(make_deps, source):
    """The order matters more than the message: a rejected query must not run.

    Asserting on `source.executed` rather than on the exception is deliberate —
    a guard that raised *after* executing would still raise.
    """
    deps = make_deps(src=source)
    tools = tools_for(deps)

    with pytest.raises(GuardRejection):
        tools["run_sql"]("DROP TABLE users")

    assert source.executed == []


def test_a_guard_violation_is_recorded_by_the_middleware_that_catches_it(
    make_deps, source
):
    """`run_sql` raising is only half the property — `ToolErrorMiddleware`
    needs the exception, but a rejected query still has to show up in
    `/trace`. `_SqlFailureRecorder` is the layer that writes it, since
    `run_sql` itself has no `Command` to write it into on a path that raises.
    """
    deps = make_deps(src=source)

    command = _sql_failure(deps, "DROP TABLE users")

    assert source.executed == []
    assert command.update["attempts"][0]["violations"]
    assert command.update["attempts"][0]["step_id"] == "q1"
    assert command.update["calls"] == 1
    assert "rewrite it" in _content(command).lower()


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
    tools = tools_for(deps)

    command = tools["run_sql"]("SELECT id, spend FROM users")

    assert "a@b.com" not in _content(command)
    assert command.update["frame"] is not None


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

    others = [
        *build_report_tools(deps),
        *build_memory_tools(deps),
        *build_schema_tool(deps),
        build_analyst_tools(deps)[1],  # lookup_definitions
    ]

    for t in others:
        body = inspect.getsource(t.func)
        assert "source.execute" not in body, f"{t.name} queries the warehouse"


def test_a_failed_query_is_recorded_and_repaired(make_deps):
    """`run_sql` itself still just raises `QuerySyntaxError` when called
    directly — that is what lets `_SqlFailureRecorder` (standing where
    `ToolErrorMiddleware(on_error=describe_failure)` used to sit in
    `analyst_middleware`) catch it. Driving the same failure through that
    middleware is what proves the recording: the attempt, the event and the
    `calls` increment are written, and the model gets a repair message back
    instead of a dead turn.
    """
    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})}, failing={"bad"})
    deps = make_deps(src=source)
    tools = tools_for(deps)

    with pytest.raises(QuerySyntaxError):
        tools["run_sql"]("SELECT bad FROM users")

    command = _sql_failure(deps, "SELECT bad FROM users")

    assert command.update["attempts"][0]["error"]
    assert command.update["attempts"][0]["sql"] == "SELECT bad FROM users"
    assert "frame" not in command.update, "a failed attempt writes no frame"
    assert command.update["calls"] == 1
    assert "fix it" in _content(command).lower()


def test_the_last_successful_query_is_the_one_kept(make_deps):
    """A turn that fails, repairs and succeeds is scored on the query that ran.

    Calling `.func` directly, as this does, bypasses `_SqlFailureRecorder`
    (see `test_a_failed_query_is_recorded_and_repaired` for what that layer
    records), so the failed call here still just raises and there is only
    ever one `Command` to inspect: the second call's. Checking that its
    `executed_sql` and `frame` reflect the query that actually ran — not the
    one that failed — is read straight off the tool's own return.
    """
    source = FakeSource(
        frames={"default": pd.DataFrame({"n": [7]})}, failing={"broken"}
    )
    deps = make_deps(src=source)
    tools = tools_for(deps)

    with pytest.raises(QuerySyntaxError):
        tools["run_sql"]("SELECT broken FROM orders")
    command = tools["run_sql"]("SELECT n FROM orders")

    assert "broken" not in command.update["executed_sql"]
    assert command.update["frame"]["rows"] == [[7]]


def test_an_empty_result_carries_the_hint_the_graph_spent_a_call_on(make_deps):
    """Zero rows raises nothing, so no retry middleware reacts to it."""
    source = FakeSource(frames={"default": pd.DataFrame()}, empty_for={"Levis"})
    deps = make_deps(src=source)
    tools = tools_for(deps)

    command = tools["run_sql"]("SELECT id FROM products WHERE brand = 'Levis'")

    assert EMPTY_HINT.strip() in _content(command)


def test_an_all_null_aggregate_counts_as_empty(make_deps):
    """`SUM(x)` over no rows returns one row of NULL, not zero rows.

    Checking `row_count == 0` alone misses the case the hint exists for.
    """
    source = FakeSource(
        frames={"default": pd.DataFrame()}, null_aggregate_for={"Levis"}
    )
    deps = make_deps(src=source)
    tools = tools_for(deps)

    command = tools["run_sql"](
        "SELECT SUM(sale_price) AS total_revenue FROM order_items WHERE brand = 'Levis'"
    )

    assert EMPTY_HINT.strip() in _content(command)


def test_a_capped_result_says_so(make_deps):
    """A sample read as a complete answer is how "20 loyal customers" was
    reported against a true 5,823."""
    source = FakeSource(
        frames={"default": pd.DataFrame({"id": [1, 2]})}, total_rows=5_823
    )
    deps = make_deps(src=source)
    tools = tools_for(deps)

    command = tools["run_sql"]("SELECT id FROM users")
    rendered = _content(command)

    assert "SAMPLE" in rendered
    assert "5823" in rendered or "5,823" in rendered


def test_lookup_definitions_says_so_when_nothing_covers_the_term(make_deps):
    """An empty string reads to a model as a definition of nothing."""
    deps = make_deps()
    tools = tools_for(deps)

    command = tools["lookup_definitions"]("who is loyal?")

    assert "Decide for yourself" in _content(command)
