"""Wiring the ReAct arm to the eval seams.

`runner.run_case` takes `ask` and `execute` and knows nothing about which agent
answered. That is the whole reason a second arm is cheap: this module supplies
the same pair `harness.build_seams` does, and the scoring is identical by
construction rather than by discipline.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from retail_agent.agent.deps import AgentDeps
from retail_agent.baseline.capture import ResultCapture
from retail_agent.baseline.react import build_react_agent
from retail_agent.evals.runner import AgentAnswer
from retail_agent.evals.usage import UsageCollector
from retail_agent.llm.messages import message_text

# The ReAct arm has no router — it only ever does analysis, because the eval
# cases are analytical. Reported as a constant rather than left blank so the
# comparison table lines up column for column with the graph arm.
INTENT = "analyze"


def answer_from_react(
    text: str,
    capture: ResultCapture,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> AgentAnswer:
    """Reduce a finished ReAct turn to what an eval can score.

    Mirrors `answer_from_state` field for field. The number comes from the
    captured frame, never from `text`: parsing prose would measure how the model
    phrased itself, which legitimately varies between model versions, rather
    than what the query returned.
    """
    frame = capture.frame

    return AgentAnswer(
        text=text,
        rows=[list(row) for row in frame.rows] if frame else [],
        columns=tuple(frame.columns) if frame else (),
        sql=capture.executed_sql,
        intent=INTENT,
        trios=capture.trio_ids,
        truncated=bool(frame.truncated) if frame else False,
        row_count=int(frame.row_count) if frame else 0,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        calls=capture.calls,
    )


def ask_once(
    deps: AgentDeps,
    question: str,
    *,
    usage: UsageCollector | None = None,
) -> AgentAnswer:
    """One ReAct turn, reduced to a scorable answer.

    A fresh capture per call, not a shared one: eval cases run sequentially
    today, but a shared capture would silently score case 3's rows as case 4's
    answer the moment a case stopped querying at all.
    """
    capture = ResultCapture()
    if usage is not None:
        usage.reset()

    agent = build_react_agent(deps, capture)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [usage]} if usage is not None else None,
    )

    return answer_from_react(
        _final_text(result),
        capture,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
    )


def _final_text(result: dict) -> str:
    """The agent's closing message.

    The last message with actual content: a run that ends on a tool call, or on
    an empty assistant turn, would otherwise report an empty answer and the PII
    scan would have nothing to look at.
    """
    for message in reversed(result.get("messages", [])):
        text = message_text(message).strip()
        if text:
            return text
    return ""


def build_react_seams(settings, *, user: str = "eval"):
    """The `ask` and `execute` pair `run_case` needs, bound to real services.

    Mirrors `harness.build_seams`. Imported lazily by the CLI so that
    `retail-agent eval --help` does not pay for BigQuery and langchain.
    """
    from retail_agent.bootstrap import build_deps
    from retail_agent.datasources.bigquery import BigQuerySource
    from retail_agent.llm.provider import build_llm

    source = BigQuerySource(settings)
    deps = build_deps(settings, llm=build_llm(settings), source=source)
    usage = UsageCollector()

    def ask(question: str) -> AgentAnswer:
        return ask_once(deps, question, usage=usage)

    def execute(sql: str) -> Sequence[Sequence[Any]]:
        # The same implementation the graph arm uses. The reference query is the
        # ruler, and two arms measured against different rulers are not being
        # compared to each other.
        result = source.execute(sql)
        return [list(row) for row in result.rows.itertuples(index=False)]

    return ask, execute
