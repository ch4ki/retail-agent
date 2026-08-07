"""The graph, not the CLI, records what it learns about a user.

Recording used to live in `cli/chat.py`, so `AgentDeps` carried a `signals`
field no node read and anything that was not the CLI — Studio, the eval harness,
a future server — learned nothing. Detection now happens inside the router's own
model call, so recording belongs in the same node.
"""

from __future__ import annotations

import pytest

from retail_agent.agent.nodes.route import RouteDecision, route_node
from retail_agent.agent.state import fresh_scratch, new_turn_state
from retail_agent.store.learning import InMemorySignalStore


def asking(question: str):
    state = new_turn_state(user_id="dana", session_id="s1", question=question)
    state.update(fresh_scratch(repair_budget=2))
    return state


def test_a_preference_is_recorded_by_the_node(make_deps):
    deps = make_deps(
        [
            RouteDecision(
                intent="analyze",
                style_field="depth",
                style_value="summary",
                style_evidence="cut to the chase",
            )
        ]
    )
    deps = deps.__class__(**{**deps.__dict__, "signals": InMemorySignalStore()})

    route_node(asking("cut to the chase, how many brands?"), deps)

    assert deps.signals.counts(user_id="dana") == {
        ("depth", "summary"): (1, "cut to the chase")
    }


def test_a_question_about_the_data_records_nothing(make_deps):
    """The non-regression that matters. "why are users in state X
    underspending?" is the brief's own example — a question about the data, not
    a request for a longer answer. Recording it would make the agent slowly
    conclude every analyst wants essays."""
    deps = make_deps([RouteDecision(intent="analyze")])
    deps = deps.__class__(**{**deps.__dict__, "signals": InMemorySignalStore()})

    route_node(asking("why are users in state X underspending?"), deps)

    assert deps.signals.counts(user_id="dana") == {}


def test_an_invented_quote_is_not_recorded(make_deps):
    """The model can hallucinate a preference; it cannot be allowed to have the
    hallucination quoted back at the user as something they said."""
    deps = make_deps(
        [
            RouteDecision(
                intent="analyze",
                style_field="depth",
                style_value="summary",
                style_evidence="keep it short",
            )
        ]
    )
    deps = deps.__class__(**{**deps.__dict__, "signals": InMemorySignalStore()})

    route_node(asking("how many brands do we carry?"), deps)

    assert deps.signals.counts(user_id="dana") == {}


def test_routing_still_works_when_there_is_no_signal_store(make_deps):
    """Learning is optional; answering is not."""
    deps = make_deps([RouteDecision(intent="analyze")])
    deps = deps.__class__(**{**deps.__dict__, "signals": None})

    assert route_node(asking("how many brands?"), deps)["intent"] == "analyze"


def test_a_failing_store_never_costs_the_turn(make_deps):
    """A learning write is worth less than an answer, always."""

    class Broken(InMemorySignalStore):
        def record(self, **kwargs):
            raise RuntimeError("database is gone")

    deps = make_deps(
        [
            RouteDecision(
                intent="analyze",
                style_field="depth",
                style_value="summary",
                style_evidence="cut to the chase",
            )
        ]
    )
    deps = deps.__class__(**{**deps.__dict__, "signals": Broken()})

    assert route_node(asking("cut to the chase"), deps)["intent"] == "analyze"


def test_a_failed_router_call_records_nothing(make_deps):
    """Degrading to `analyze` must not also invent a preference."""

    class Exploding:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages, **kwargs):
            raise RuntimeError("provider down")

    deps = make_deps([])
    deps = deps.__class__(
        **{**deps.__dict__, "llm": Exploding(), "signals": InMemorySignalStore()}
    )

    result = route_node(asking("cut to the chase"), deps)

    assert result["intent"] == "analyze"
    assert deps.signals.counts(user_id="dana") == {}
