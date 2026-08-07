"""A truncated result must never be reported as a complete one.

The guard caps every query so a result set stays printable and affordable. That
cap is invisible in the result: 500 rows returned looks identical whether there
were 500 or 5,823. Measured on the real warehouse, the query the agent wrote for
"how many loyal customers do we have?" matched 5,823 rows and the agent saw 500
— then narrated a number derived from what it could see.

So the cap has to announce itself. Nothing here stops the truncation; it makes
it impossible to mistake for the whole answer.
"""

from __future__ import annotations

import pandas as pd

from retail_agent.agent.nodes.execute import execute_node
from retail_agent.agent.nodes.synthesize import synthesize_node
from retail_agent.agent.state import (
    AnalysisStep,
    MaskedFrame,
    fresh_scratch,
    new_turn_state,
)
from tests.component.conftest import FakeSource


def state_for(sql: str, question: str = "how many loyal customers?"):
    state = new_turn_state(user_id="dana", session_id="s1", question=question)
    state.update(fresh_scratch(repair_budget=2))
    state["plan"] = [AnalysisStep(id="step_1", question=question, sql=sql)]
    return state


def rows(n: int) -> pd.DataFrame:
    return pd.DataFrame({"user_id": list(range(n))})


def test_a_result_that_fills_the_limit_is_marked_truncated(make_deps):
    sql = "SELECT user_id FROM orders LIMIT 500"
    source = FakeSource(frames={"orders": rows(500)})
    deps = make_deps([], src=source)

    result = execute_node(state_for(sql), deps)

    assert result["frames"]["step_1"].truncated


def test_a_result_below_the_limit_is_not_marked(make_deps):
    """The common case. Marking everything would make the warning noise."""
    sql = "SELECT user_id FROM orders LIMIT 500"
    deps = make_deps([], src=FakeSource(frames={"orders": rows(12)}))

    result = execute_node(state_for(sql), deps)

    assert not result["frames"]["step_1"].truncated


def test_an_aggregate_with_no_limit_is_never_marked(make_deps):
    """One row from `SELECT COUNT(*)` is the whole answer, and warning about it
    would teach the reader to ignore the warning."""
    sql = "SELECT COUNT(*) AS n FROM orders"
    deps = make_deps([], src=FakeSource(frames={"orders": pd.DataFrame({"n": [5823]})}))

    result = execute_node(state_for(sql), deps)

    assert not result["frames"]["step_1"].truncated


def test_the_synthesizer_is_told_the_rows_are_partial(make_deps):
    """Otherwise it counts what it can see. That is exactly how "20 customers
    are classified as loyal" was produced against a true 5,823."""
    deps = make_deps(["Answer."])
    state = state_for("SELECT user_id FROM orders LIMIT 500")
    state["frames"] = {
        "step_1": MaskedFrame(
            columns=("user_id",),
            rows=((1,), (2,)),
            row_count=500,
            redactions=0,
            truncated=True,
        )
    }

    synthesize_node(state, deps)

    prompt = "\n".join(deps.llm.prompts).lower()
    assert "truncated" in prompt or "partial" in prompt
    assert "500" in prompt


def test_a_complete_result_carries_no_warning(make_deps):
    deps = make_deps(["Answer."])
    state = state_for("SELECT COUNT(*) AS n FROM orders")
    state["frames"] = {
        "step_1": MaskedFrame(columns=("n",), rows=((5823,),), row_count=1, redactions=0)
    }

    synthesize_node(state, deps)

    assert "truncated" not in "\n".join(deps.llm.prompts).lower()


def test_the_sql_writer_is_told_to_aggregate_in_the_query(make_deps):
    """Prevention rather than detection: a counting question answered with
    `SELECT COUNT(*)` returns one row and cannot be truncated at all."""
    from retail_agent.agent.nodes.sql import draft_sql_node

    deps = make_deps(["SELECT COUNT(*) AS n FROM orders"])

    draft_sql_node(state_for(sql=None, question="how many loyal customers?"), deps)

    prompt = "\n".join(deps.llm.prompts).lower()
    assert "count(" in prompt and "truncat" in prompt
