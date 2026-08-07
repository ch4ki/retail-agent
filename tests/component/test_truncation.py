"""A truncated result must never be reported as a complete one.

Results are capped when they are read, not by a LIMIT in the SQL, so
`row_count` is the true size of the result even when only some rows were
fetched. That makes "was this truncated" an exact comparison rather than a guess
from the query text — and it means the count is available and correct even when
the agent returned rows instead of an aggregate.

Nothing here stops the truncation. It makes it impossible to mistake a sample
for the whole answer.
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


def test_more_rows_matched_than_were_fetched_is_marked_truncated(make_deps):
    """Exact: the warehouse reports 5,823 matched while 500 were fetched."""
    source = FakeSource(frames={"orders": rows(500)}, total_rows=5823)
    deps = make_deps([], src=source)

    result = execute_node(state_for("SELECT user_id FROM orders"), deps)

    frame = result["frames"]["step_1"]
    assert frame.truncated
    assert frame.row_count == 5823, "the true total, not the number fetched"


def test_a_complete_result_is_not_marked(make_deps):
    """The common case. Marking everything would make the warning noise."""
    deps = make_deps([], src=FakeSource(frames={"orders": rows(12)}))

    result = execute_node(state_for("SELECT user_id FROM orders"), deps)

    assert not result["frames"]["step_1"].truncated


def test_a_result_exactly_the_size_of_the_fetch_is_not_marked(make_deps):
    """The old rule guessed from the SQL's LIMIT and had to call this truncated
    just in case. The warehouse now says 500 matched and 500 were fetched, so
    the hedge is gone."""
    deps = make_deps([], src=FakeSource(frames={"orders": rows(500)}, total_rows=500))

    result = execute_node(state_for("SELECT user_id FROM orders"), deps)

    assert not result["frames"]["step_1"].truncated


def test_an_aggregate_with_no_limit_is_never_marked(make_deps):
    """One row from `SELECT COUNT(*)` is the whole answer, and warning about it
    would teach the reader to ignore the warning."""
    deps = make_deps([], src=FakeSource(frames={"orders": pd.DataFrame({"n": [5823]})}))

    result = execute_node(state_for("SELECT COUNT(*) AS n FROM orders"), deps)

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
            row_count=5823,
            redactions=0,
            truncated=True,
        )
    }

    synthesize_node(state, deps)

    prompt = "\n".join(deps.llm.prompts).lower()
    assert "sample" in prompt or "partial" in prompt
    # The exact total, which is the number the question actually wants. The old
    # message could only say "first 500" — the cap, not the answer.
    assert "5823" in prompt


def test_a_complete_result_carries_no_warning(make_deps):
    deps = make_deps(["Answer."])
    state = state_for("SELECT COUNT(*) AS n FROM orders")
    state["frames"] = {
        "step_1": MaskedFrame(columns=("n",), rows=((5823,),), row_count=1, redactions=0)
    }

    synthesize_node(state, deps)

    assert "sample" not in "\n".join(deps.llm.prompts).lower()


def test_the_sql_writer_is_told_to_aggregate_in_the_query(make_deps):
    """Prevention rather than detection: a counting question answered with
    `SELECT COUNT(*)` returns one row and cannot be truncated at all."""
    from retail_agent.agent.nodes.sql import draft_sql_node

    deps = make_deps(["SELECT COUNT(*) AS n FROM orders"])

    draft_sql_node(state_for(sql=None, question="how many loyal customers?"), deps)

    assert "count(" in "\n".join(deps.llm.prompts).lower()
