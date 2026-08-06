"""Asking what a term means, once, then remembering the answer.

Same shape as the delete confirmation: the pause is a breakpoint declared in
graph.py, and the node that acts on the reply sits on the far side of it.
"""

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from retail_agent.agent.graph import build_graph
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.store.definitions import InMemoryDefinitionStore
from tests.component.conftest import FakeSource

ROWS = pd.DataFrame({"n": [42]})

ANALYSIS = [
    {"intent": "analyze"},
    {"steps": ["count customers meeting the rule"]},
    "SELECT COUNT(*) AS n FROM orders",
    "There are 42.",
]


def start(deps, question="which customers are at risk?"):
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s1"}}
    state = graph.invoke(
        {"messages": [HumanMessage(content=question)], "user_id": "dana",
         "session_id": "s1"},
        config,
    )
    return graph, config, state


def with_stores(deps, definitions):
    return deps.__class__(
        **{**deps.__dict__, "trios": list(SEED_TRIOS), "definitions": definitions}
    )


def test_an_undefined_term_pauses_before_any_query_runs(make_deps):
    source = FakeSource(frames={"ok": ROWS})
    deps = with_stores(make_deps(ANALYSIS, src=source), InMemoryDefinitionStore())

    graph, config, _ = start(deps)

    assert graph.get_state(config).next == ("await_definition",)
    assert graph.get_state(config).values["pending_term"] == "at risk"
    assert source.executed == [], "nothing ran before the user was asked"


def test_the_answer_is_used_and_remembered(make_deps):
    definitions = InMemoryDefinitionStore()
    deps = with_stores(make_deps(ANALYSIS, src=FakeSource(frames={"ok": ROWS})), definitions)
    graph, config, _ = start(deps)

    graph.update_state(
        config, {"definition_reply": "no order in 60 days"}, as_node="await_definition"
    )
    final = graph.invoke(None, config)

    assert definitions.lookup(user_id="dana", term="at risk").definition == (
        "no order in 60 days"
    )
    assert final["status"] == "ok"


def test_a_remembered_term_is_never_asked_about_again(make_deps):
    """The whole point. Asking the same person what "at risk" means every week
    is how a safety feature becomes an irritation."""
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="dana", term="at risk", definition="no order in 60 days")
    deps = with_stores(make_deps(ANALYSIS, src=FakeSource(frames={"ok": ROWS})), definitions)

    graph, config, state = start(deps)

    assert graph.get_state(config).next == (), "answered without pausing"
    assert state["status"] == "ok"


def test_a_remembered_definition_reaches_the_sql_prompt(make_deps):
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="dana", term="at risk", definition="no order in 60 days")
    deps = with_stores(make_deps(ANALYSIS, src=FakeSource(frames={"ok": ROWS})), definitions)

    start(deps)

    sql_prompt = [p for p in deps.llm.prompts if "Write one BigQuery" in p][0]
    assert "no order in 60 days" in sql_prompt


def test_pressing_enter_falls_back_to_assuming(make_deps):
    """Opting out must still answer. Refusing an executive's question is not
    safety, it is unhelpfulness."""
    definitions = InMemoryDefinitionStore()
    deps = with_stores(make_deps(ANALYSIS, src=FakeSource(frames={"ok": ROWS})), definitions)
    graph, config, _ = start(deps)

    graph.update_state(config, {"definition_reply": ""}, as_node="await_definition")
    final = graph.invoke(None, config)

    assert final["status"] == "ok"
    assert final["assumed_terms"] == ["at risk"], "disclosed rather than silently chosen"
    assert definitions.lookup(user_id="dana", term="at risk") is None, "nothing saved"


def test_a_term_the_corpus_defines_is_never_asked_about(make_deps):
    """The bucket outranks the question: if the analysts settled it, there is
    nothing to ask."""
    deps = with_stores(make_deps(ANALYSIS, src=FakeSource(frames={"ok": ROWS})), InMemoryDefinitionStore())

    graph, config, _ = start(deps, "why did our churn rate spike?")

    assert graph.get_state(config).next == ()


def test_a_question_with_no_business_terms_never_pauses(make_deps):
    deps = with_stores(make_deps(ANALYSIS, src=FakeSource(frames={"ok": ROWS})), InMemoryDefinitionStore())

    graph, config, _ = start(deps, "what was total revenue in March 2024?")

    assert graph.get_state(config).next == ()
