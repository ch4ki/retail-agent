"""The planner has to know what the business terms mean.

Measured live: asked "how many engaged customers", the planner emitted

    Retrieve from users the count of distinct users who have placed at least
    one order with status 'Complete' ...

"engaged" is gone, replaced by the planner's own invented criterion. `draft_sql`
then works from that sentence, and the definitions block that arrives with it is
useless — the term it defines is no longer in the question being answered. The
graph returned 8,067 against an expected 3,450 for exactly this reason.

`recall` runs before `plan` and has already resolved the definitions. They were
simply never passed on.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from retail_agent.agent.nodes.plan import plan_node
from retail_agent.knowledge.trios import Trio

from .conftest import FakeSource

LOYAL = Trio(
    id="loyal_v1",
    question="who are our loyal customers?",
    sql="SELECT 1",
    report="...",
    metric_definitions={"loyal": "three or more completed orders, all time"},
)


@pytest.fixture
def source():
    return FakeSource(frames={"default": pd.DataFrame({"n": [1]})})


def planner_prompt(deps, question: str) -> str:
    """Run the planner and return the prompt it actually sent."""
    state = {
        "messages": [HumanMessage(content=question)],
        "trio_ids": [LOYAL.id],
    }
    plan_node(state, deps)
    return deps.llm.prompts[-1]


def test_the_planner_is_told_what_the_business_terms_mean(make_deps, source):
    """The definition must be in the planner's prompt, not only the SQL node's.
    The planner decides the shape of the answer; a shape chosen without the
    definition cannot be corrected downstream."""
    deps = replace(
        make_deps([{"steps": ["count loyal customers"]}], src=source), trios=[LOYAL]
    )

    assert "three or more completed orders" in planner_prompt(deps, "how many loyal customers?")


def test_the_planner_is_instructed_to_keep_the_term_in_the_step(make_deps, source):
    """Passing the definition is not enough on its own — the planner has to be
    told to carry the term through rather than paraphrase it away."""
    from retail_agent.agent.prompts import PLANNER_PROMPT

    assert "{definitions}" in PLANNER_PROMPT
    lowered = PLANNER_PROMPT.lower()
    assert "verbatim" in lowered or "exactly as the user wrote" in lowered


def test_a_question_with_no_business_term_still_plans(make_deps, source):
    """The common case must not regress: no trios retrieved, empty definitions,
    and the planner still decomposes normally."""
    deps = replace(make_deps([{"steps": ["total revenue in 2023"]}], src=source), trios=[])

    state = {"messages": [HumanMessage(content="what was revenue in 2023?")]}
    out = plan_node(state, deps)

    assert [s.question for s in out["plan"]] == ["total revenue in 2023"]


def test_duplicate_steps_are_collapsed(make_deps, source):
    """Observed live: the planner returned the same sentence twice for one
    question, which runs the same query twice and lets `_final_frame` score a
    duplicate as though it were a second finding."""
    deps = replace(
        make_deps(
            [{"steps": ["count engaged customers", "count engaged customers"]}],
            src=source,
        ),
        trios=[LOYAL],
    )

    out = plan_node({"messages": [HumanMessage(content="how many engaged?")]}, deps)

    assert [s.question for s in out["plan"]] == ["count engaged customers"]
