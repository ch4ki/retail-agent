"""Every turn is recorded by the graph, not by whoever called it.

`traces.record(...)` used to live in `cli/chat.py`, so whether a turn was
remembered depended on who invoked the graph rather than on what the graph did.
Studio invokes the compiled graph object directly and the eval harness calls
`run_turn`; neither passed through that line, so neither left a trace. Measured
before this change: a turn driven straight through the graph recorded zero.

`finish_turn` sits on every path out, which works because the graph already
guarantees that every path ends.
"""

from __future__ import annotations

import pandas as pd

from retail_agent.agent.graph import build_graph, run_turn
from tests.component.conftest import FakeSource


def turn(graph, question="top customers by spend"):
    return run_turn(graph, user_id="dana", session_id="s1", question=question)


def analysis(source):
    return [
        {"intent": "analyze"},
        {"steps": ["total spend per customer"]},
        "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",
        "Your top customer spent $100.",
    ]


def test_an_analysis_turn_is_recorded(make_deps, source):
    deps = make_deps(analysis(source), src=source)

    turn(build_graph(deps))

    assert len(deps.traces.recent(owner_id="dana")) == 1


def test_a_schema_turn_is_recorded(make_deps, source):
    """A path that runs no SQL is still a turn someone may ask about."""
    deps = make_deps([{"intent": "schema"}, "We hold orders and customers."], src=source)

    turn(build_graph(deps), "what data do you have?")

    assert len(deps.traces.recent(owner_id="dana")) == 1


def test_a_chat_turn_is_recorded(make_deps, source):
    deps = make_deps([{"intent": "chat"}, "Glad that helped."], src=source)

    turn(build_graph(deps), "thanks")

    assert len(deps.traces.recent(owner_id="dana")) == 1


def test_a_report_operation_is_recorded(make_deps, source):
    deps = make_deps(
        [{"intent": "report_op"}, {"operation": "list", "match": "", "title": ""}],
        src=source,
    )

    turn(build_graph(deps), "list my reports")

    assert len(deps.traces.recent(owner_id="dana")) == 1


def test_the_recorded_trace_carries_what_the_turn_did(make_deps, source):
    """A trace nobody can debug from is not worth the write."""
    deps = make_deps(analysis(source), src=source)

    state = turn(build_graph(deps))

    stored = deps.traces.get(owner_id="dana", turn_id=state["turn_id"])
    assert stored is not None
    assert stored.question == "top customers by spend"
    assert stored.events, "node timings are the point of a trace"
    assert stored.attempts, "the SQL that ran is the other point"


def test_a_failing_trace_store_never_costs_the_answer(make_deps, source):
    """A trace is a debugging aid. Losing one must not lose the user's answer,
    which is the same rule the CLI applied before this moved."""

    class Broken:
        def record(self, trace):
            raise RuntimeError("database is gone")

        def recent(self, **kwargs):
            return []

        def get(self, **kwargs):
            return None

        def metrics(self, **kwargs):
            return {}

    deps = make_deps(analysis(source), src=source)
    deps = deps.__class__(**{**deps.__dict__, "traces": Broken()})

    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert "$100" in state["answer"]


def test_each_turn_is_recorded_once(make_deps, source):
    """Two turns, two rows — not one row twice or three rows for two turns."""
    deps = make_deps(
        [{"intent": "chat"}, "One.", {"intent": "chat"}, "Two."], src=source
    )
    graph = build_graph(deps)

    turn(graph, "first")
    turn(graph, "second")

    assert len(deps.traces.recent(owner_id="dana")) == 2
