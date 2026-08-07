"""A whole ReAct turn, offline.

`create_agent` is driven by a scripted chat model here, so the loop, the
middleware and the tools all really run — only the provider and the warehouse
are fake. This is the test that would catch the tools being wired to a capture
the seam never reads, which no amount of unit testing the parts would find.
"""

from __future__ import annotations

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from retail_agent.baseline.seams import ask_once

from .conftest import FakeSource


class ToolCallingFake(FakeMessagesListChatModel):
    """A fake that accepts `bind_tools`.

    `create_agent` binds tools to the model before the first call. The stock
    fakes raise `NotImplementedError` there, which would fail the turn before
    any of the code under test ran.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def call(name: str, **args) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"c_{name}"}],
    )


@pytest.fixture
def source():
    return FakeSource(frames={"default": pd.DataFrame({"total": [1234.5]})})


def test_a_turn_that_queries_once_reports_the_rows_it_got(make_deps, source):
    model = ToolCallingFake(
        responses=[
            call("run_sql", sql="SELECT SUM(sale_price) AS total FROM order_items"),
            AIMessage(content="Revenue was 1,234.50."),
        ]
    )
    deps = _with_model(make_deps([], src=source), model)

    answer = ask_once(deps, "What was total revenue?")

    assert answer.rows == [[1234.5]]
    assert answer.columns == ("total",)
    assert "order_items" in answer.sql
    assert answer.text == "Revenue was 1,234.50."
    assert answer.calls == 1


def test_a_turn_that_looks_up_definitions_first_records_the_trio(make_deps, source):
    """The elective path actually being taken. If the model chooses to consult
    the corpus, the answer says which trio it consulted — the same field the
    graph fills from `trio_ids`."""
    from dataclasses import replace

    from retail_agent.knowledge.trios import Trio

    trio = Trio(
        id="loyal_v1",
        question="who is loyal?",
        sql="SELECT 1",
        report="...",
        metric_definitions={"loyal": "three or more orders in the trailing year"},
    )
    model = ToolCallingFake(
        responses=[
            call("lookup_definitions", question="how many loyal customers?"),
            call("run_sql", sql="SELECT COUNT(*) AS total FROM users"),
            AIMessage(content="1,234 loyal customers."),
        ]
    )
    deps = replace(_with_model(make_deps([], src=source), model), trios=[trio])

    answer = ask_once(deps, "How many loyal customers?")

    assert answer.trios == ("loyal_v1",)
    assert answer.calls == 2


def test_a_turn_that_never_queries_scores_as_no_rows(make_deps, source):
    """The model answering from its own priors without touching the warehouse.
    It must score as unanswered rather than as whatever prose it produced."""
    model = ToolCallingFake(responses=[AIMessage(content="Probably about 5,000.")])
    deps = _with_model(make_deps([], src=source), model)

    answer = ask_once(deps, "How many customers are loyal?")

    assert answer.rows == []
    assert answer.sql == ""
    assert answer.text == "Probably about 5,000."


def test_two_turns_do_not_share_a_capture(make_deps, source):
    """Each case gets a fresh capture. Sharing one would let case 3's rows be
    scored as case 4's answer when case 4 never queried at all."""
    first = _with_model(
        make_deps([], src=source),
        ToolCallingFake(
            responses=[
                call("run_sql", sql="SELECT SUM(sale_price) AS total FROM order_items"),
                AIMessage(content="1,234.50"),
            ]
        ),
    )
    ask_once(first, "What was revenue?")

    second = _with_model(
        make_deps([], src=source),
        ToolCallingFake(responses=[AIMessage(content="I cannot answer that.")]),
    )
    answer = ask_once(second, "What is our churn rate?")

    assert answer.rows == []


def _with_model(deps, model):
    from dataclasses import replace

    return replace(deps, llm=model)
