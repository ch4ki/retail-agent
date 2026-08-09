"""The analyst subagent: what it refuses to do, and what it discloses.

`recall_node` ran unconditionally on every analyze turn and the graph branched
on what it found. Both of those properties now live in this wrapper, before the
subagent is built — which is the point. A model that elected not to look a term
up could not lose them here even if it wanted to.
"""

import pandas as pd

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.subagents import build_subagents
from retail_agent.knowledge.trios import Trio
from retail_agent.store.definitions import InMemoryDefinitionStore

from .conftest import FakeSource

LOYAL = Trio(
    id="trio-loyal",
    question="How many loyal customers do we have?",
    sql="SELECT 1",
    report="Loyalty is measured over a rolling year.",
    metric_definitions={"loyal": "three or more completed orders in 12 months"},
)


def subagents_for(deps, question="who are our loyal customers?"):
    capture = TurnCapture(user_id="exec", session_id="s1", question=question)
    return {fn.__name__: fn for fn in build_subagents(deps, capture)}, capture


def test_an_unsettled_term_stops_the_turn_before_anything_is_queried(make_deps, source):
    """The property the graph got from a breakpoint: no spend before meaning.

    Asserting on `source.executed` rather than on the returned text — a wrapper
    that queried first and then asked would return the same sentence.
    """
    deps = make_deps(src=source, definitions=InMemoryDefinitionStore())
    analyst, capture = subagents_for(deps)

    answer = analyst["analyst"]("who are our loyal customers?")

    assert source.executed == []
    assert "loyal" in answer
    assert "assume_undefined" in answer
    assert capture.attempts == []


def test_a_trio_settles_the_term_and_the_query_runs(make_deps):
    """With the corpus covering it, nothing is asked and the analysis proceeds."""
    source = FakeSource(frames={"default": pd.DataFrame({"loyal_customers": [42]})})
    deps = make_deps(
        script=[
            [("run_sql", {"sql": "SELECT COUNT(*) AS loyal_customers FROM users"})],
            "42 customers are loyal.",
        ],
        src=source,
        definitions=InMemoryDefinitionStore(),
        trios=[LOYAL],
    )
    analyst, capture = subagents_for(deps)

    answer = analyst["analyst"]("who are our loyal customers?")

    assert "42" in answer
    assert capture.trio_ids == ["trio-loyal"]
    assert source.executed


def test_the_agreed_definition_reaches_the_model(make_deps):
    """Retrieval that never reaches the prompt is retrieval that did nothing."""
    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})})
    deps = make_deps(script=["one."], src=source, trios=[LOYAL])
    analyst, _ = subagents_for(deps)

    analyst["analyst"]("who are our loyal customers?")

    assert any(
        "three or more completed orders" in prompt for prompt in deps.llm.prompts
    )


def test_assuming_a_definition_forces_the_assumption_into_the_answer(make_deps):
    """The number is only trustworthy if the reader knows which judgement made it.

    The note is appended by the wrapper rather than requested in the prompt, so
    a model that ignores the instruction still cannot return the figure alone.
    """
    source = FakeSource(frames={"default": pd.DataFrame({"n": [9]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "9."],
        src=source,
        definitions=InMemoryDefinitionStore(),
    )
    analyst, capture = subagents_for(deps)

    answer = analyst["analyst"]("who are our loyal customers?", assume_undefined=True)

    assert "no agreed definition" in answer.lower()
    assert capture.assumed_terms == ["loyal"]


def test_without_a_definition_store_the_agent_assumes_rather_than_asks(make_deps):
    """Asking is only worth it if the answer can be kept.

    Without somewhere to remember it the agent would ask the same person the
    same question every turn, which is worse than assuming and saying so.
    """
    source = FakeSource(frames={"default": pd.DataFrame({"n": [3]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "3."],
        src=source,
    )
    analyst, capture = subagents_for(deps)

    answer = analyst["analyst"]("who are our loyal customers?")

    assert "3" in answer
    assert capture.assumed_terms == ["loyal"]


def test_the_report_writer_is_shown_how_analysts_here_write(make_deps):
    """The other half of what a trio carries, and it had gone missing.

    `metric_definitions` says what to measure and reaches the analyst. `report`
    demonstrates the house shape — split by cohort, compare against a baseline,
    close with numbered actions — which is hard to specify and easy to show. It
    was injected by the graph's synthesis node, and deleting that node dropped
    it silently: nothing failed, the corpus field simply stopped being read.
    """
    deps = make_deps(script=["## Summary\nRevenue rose."], trios=[LOYAL])
    writer, capture = subagents_for(deps)
    capture.record_definitions(["trio-loyal"])

    writer["report_writer"]("Revenue rose 4% in Q1.")

    assert "Loyalty is measured over a rolling year." in deps.llm.prompts[0]


def test_a_report_with_no_trio_behind_it_still_writes(make_deps):
    """An empty corpus is a valid state; the examples block just goes away."""
    deps = make_deps(script=["## Summary\nRevenue rose."])
    writer, _ = subagents_for(deps)

    assert writer["report_writer"]("Revenue rose 4%.")


def test_the_report_writer_runs_through_the_provider_chain(make_deps):
    """The tool-less subagent compiles down a different path, and it was broken.

    Every other agent here has tools, so every other agent compiled through
    `bind_tools`, which the chain implements. `create_agent` binds a tool-less
    agent with `bind` instead — so the first live report died on
    `AttributeError` while the whole offline suite stayed green, because the
    doubles are handed to `create_agent` directly rather than through a chain.
    """
    from retail_agent.llm.resilience import ResilientChatModel

    deps = make_deps(script=["## Summary\nRevenue rose."])
    chain = ResilientChatModel([("primary", deps.llm), ("fallback", deps.llm)])
    object.__setattr__(deps, "llm", chain)
    writer, _ = subagents_for(deps)

    assert "Revenue rose" in writer["report_writer"]("Revenue rose 4% in Q1.")


def test_the_report_writer_cannot_reach_the_data(make_deps):
    """No tools, so a number missing from the brief cannot appear in the report."""
    deps = make_deps(script=["## Summary\nRevenue rose."])
    writer, _ = subagents_for(deps)

    body = writer["report_writer"]("Revenue rose 4% in Q1.")

    assert "Revenue rose" in body
    assert deps.llm.bound_tools == []
