"""The analyst subagent: what it is given, and what it discloses.

`recall_node` ran unconditionally on every analyze turn, and that still holds:
retrieval happens in this wrapper, before the subagent is built, so a model
that elected not to look a term up cannot lose the corpus.

What this no longer does is *decide* whether a term is unsettled. That was a
regex over nineteen words, and it is gone — `ask_for_definitions` asks and
records, and the analyst reads what it recorded. The disclosure is still forced
here rather than requested in a prompt, which is the property worth keeping.
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


def test_the_executives_own_definitions_reach_the_model(make_deps):
    """All of them, not the ones a regex picked out of the question. The
    filtering step that needed a term list is gone, and the whole set costs one
    read — so a word the old detector never recognised is now in front of the
    model anyway."""
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="exec", term="LGB", definition="low gross basket")
    source = FakeSource(frames={"default": pd.DataFrame({"n": [9]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "9."],
        src=source,
        definitions=definitions,
    )
    analyst, _ = subagents_for(deps)

    analyst["analyst"]("how many LGB customers?")

    assert any("low gross basket" in prompt for prompt in deps.llm.prompts)


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


def test_a_recorded_assumption_is_forced_into_the_answer(make_deps):
    """The number is only trustworthy if the reader knows which judgement made it.

    The note is appended by the wrapper rather than requested in the prompt, so
    a model that ignores the instruction still cannot return the figure alone.
    The terms come from the capture — written earlier in the turn by
    `ask_for_definitions` when nobody was there to answer it.
    """
    source = FakeSource(frames={"default": pd.DataFrame({"n": [9]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "9."],
        src=source,
        definitions=InMemoryDefinitionStore(),
    )
    analyst, capture = subagents_for(deps)
    capture.record_assumptions(["LGB"])

    answer = analyst["analyst"]("how many LGB customers?")

    assert "no agreed definition" in answer.lower()
    assert "LGB" in answer


def test_nothing_assumed_means_no_disclosure(make_deps):
    """A caveat on a question that did not need one is noise, and noise is how
    a warning stops being read."""
    source = FakeSource(frames={"default": pd.DataFrame({"n": [3]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "3."],
        src=source,
        definitions=InMemoryDefinitionStore(),
    )
    analyst, _ = subagents_for(deps)

    answer = analyst["analyst"]("how much revenue did we make in March?")

    assert "3" in answer
    assert "no agreed definition" not in answer.lower()


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

    writer["report_writer"]("Revenue rose 4% in Q1.", title="Q1 Revenue")

    assert "Loyalty is measured over a rolling year." in deps.llm.prompts[0]


def test_a_report_with_no_trio_behind_it_still_writes(make_deps):
    """An empty corpus is a valid state; the examples block just goes away."""
    deps = make_deps(script=["## Summary\nRevenue rose."])
    writer, _ = subagents_for(deps)

    assert writer["report_writer"]("Revenue rose 4%.", title="Q1 Revenue")


def test_the_report_writer_runs_through_the_provider_chain(make_deps):
    """The tool-less subagent compiles down a different path, and it was broken.

    Every other agent here has tools, so every other agent compiled through
    `bind_tools`. `create_agent` binds a tool-less agent with `bind` instead,
    and the chain object that used to sit in front of the model implemented one
    and not the other — so the first live report died on `AttributeError` while
    the whole offline suite stayed green.

    The chain object is gone and the fallbacks are middleware now, which is
    what makes that class of bug unreachable: middleware is handed a model
    rather than having to impersonate one. Kept, with a fallback configured, so
    the tool-less compile path stays covered.
    """
    deps = make_deps(script=["## Summary\nRevenue rose."])
    object.__setattr__(deps, "llm_fallbacks", [deps.llm])
    writer, _ = subagents_for(deps)

    receipt = writer["report_writer"]("Revenue rose 4% in Q1.", title="Q1 Revenue")

    # The receipt itself carries no report text (see test_report_tools.py) —
    # what this test needs is proof the tool-less compile path produced a
    # report at all, so it checks the store rather than the return value.
    assert "written" in receipt
    assert "Revenue rose" in deps.reports.list_reports(owner_id="exec")[0].body


def test_the_report_writer_cannot_reach_the_data(make_deps):
    """No tools, so a number missing from the brief cannot appear in the report."""
    deps = make_deps(script=["## Summary\nRevenue rose."])
    writer, _ = subagents_for(deps)

    writer["report_writer"]("Revenue rose 4% in Q1.", title="Q1 Revenue")

    assert "Revenue rose" in deps.reports.list_reports(owner_id="exec")[0].body
    assert deps.llm.bound_tools == []
