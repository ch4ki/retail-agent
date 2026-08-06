"""Regression tests for providers that return structured content blocks.

Gemini returns `[{"type": "text", "text": ..., "extras": {...}}]` where OpenAI
returns a plain string. Stringifying `.content` directly produced a Python repr
of the list, which made the router misclassify every turn and made the SQL node
emit an array literal that the guard rejected as `found Array`.
"""

import pytest

from retail_agent.agent.graph import build_graph, run_turn


def turn(graph, question):
    return run_turn(
        graph, user_id="dana", session_id="s1", question=question
    )


@pytest.mark.parametrize("blocks", [False, True], ids=["string-content", "block-content"])
def test_schema_question_routes_to_schema(make_deps, source, blocks):
    deps = make_deps(
        [{"intent": "schema"}, "We hold orders, products and customers."],
        src=source,
        blocks=blocks,
    )
    state = turn(build_graph(deps), "what data do you have?")

    assert state["intent"] == "schema"
    assert source.executed == []
    assert "orders" in state["answer"]


@pytest.mark.parametrize("blocks", [False, True], ids=["string-content", "block-content"])
def test_analysis_turn_produces_runnable_sql(make_deps, source, blocks):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["total spend per customer"]},
            "SELECT id, spend FROM users",
            "Your top customer spent $100.",
        ],
        src=source,
        blocks=blocks,
    )
    state = turn(build_graph(deps), "top customers by spend")

    assert state["status"] == "ok"
    assert len(source.executed) == 1
    assert source.executed[0].upper().startswith("SELECT")
    assert "$100" in state["answer"]


@pytest.mark.parametrize("blocks", [False, True], ids=["string-content", "block-content"])
def test_answer_is_never_a_python_repr(make_deps, source, blocks):
    deps = make_deps([{"intent": "chat"}, "Glad that helped."], src=source, blocks=blocks)
    state = turn(build_graph(deps), "thanks")

    assert state["answer"] == "Glad that helped."
    assert not state["answer"].startswith("[{")
