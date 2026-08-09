"""Wiring the agent to the eval harness.

`evals/runner.run_case` takes `ask` and `execute` and knows nothing about
LangGraph or BigQuery. This is the only place those get connected, which is why
the reduction below is a pure function tested on its own rather than something
only a live run can exercise.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.subagents import final_text
from retail_agent.agent.supervisor import build_agent
from retail_agent.evals.runner import AgentAnswer
from retail_agent.evals.usage import UsageCollector


def answer_from_capture(
    text: str,
    capture: TurnCapture,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> AgentAnswer:
    """Reduce a finished turn to what an eval can score.

    The number comes from the captured frame, never from `text`: parsing prose
    would measure how the model phrased itself, which legitimately varies
    between model versions, rather than what the query returned.
    """
    frame = capture.frame

    return AgentAnswer(
        text=text,
        rows=[list(row) for row in frame.rows] if frame else [],
        columns=tuple(frame.columns) if frame else (),
        sql=capture.executed_sql,
        intent=capture.intent,
        trios=tuple(capture.trio_ids),
        truncated=bool(frame.truncated) if frame else False,
        row_count=int(frame.row_count) if frame else 0,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        calls=capture.calls,
        # In order, not as a set: the suite scores whether the agent asked what
        # a term meant *before* it spent a query, and the sequence is the only
        # place that shows.
        tools=tuple(name for name, _, _ in capture.events),
        report_ids=tuple(r.report_id for r in capture.reports_written),
        report_bodies=tuple(r.body for r in capture.reports_written),
    )


def ask_once(
    deps: AgentDeps,
    question: str,
    *,
    user: str = "eval",
    usage: UsageCollector | None = None,
) -> AgentAnswer:
    """One turn, reduced to a scorable answer.

    A fresh capture and a fresh thread per call. Sharing either would let case
    3's rows or case 3's conversation reach case 4, and the suite would be
    measuring memory rather than analysis.
    """
    session = f"eval-{uuid.uuid4().hex[:8]}"
    capture = TurnCapture(user_id=user, session_id=session, question=question)
    if usage is not None:
        usage.reset()

    agent = build_agent(deps, capture, checkpointer=MemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": session}}
    if usage is not None:
        config["callbacks"] = [usage]

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )

    # A turn paused for a delete confirmation never produces a number. Recorded
    # as an unanswered turn rather than hanging the run.
    if getattr(agent.get_state(config), "next", ()):
        return AgentAnswer(
            text="[paused awaiting approval]",
            rows=[],
            columns=(),
            tokens_in=usage.tokens_in if usage else 0,
            tokens_out=usage.tokens_out if usage else 0,
        )

    return answer_from_capture(
        final_text(result),
        capture,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
    )


def build_seams(settings, *, user: str = "eval"):
    """The `ask` and `execute` pair `run_case` needs, bound to real services.

    Imported lazily by the CLI so that `retail-agent eval --help` does not pay
    for BigQuery and langchain.
    """
    from retail_agent.bootstrap import build_deps
    from retail_agent.datasources.bigquery import BigQuerySource
    from retail_agent.llm.provider import build_models

    source = BigQuerySource(settings)
    llm, llm_fallbacks = build_models(settings)
    deps = build_deps(
        settings, llm=llm, llm_fallbacks=llm_fallbacks, source=source
    )
    usage = UsageCollector()

    def ask(question: str) -> AgentAnswer:
        return ask_once(deps, question, user=user, usage=usage)

    def execute(sql: str) -> Sequence[Sequence[Any]]:
        # The reference query bypasses the guard and the agent entirely: it is
        # the ruler, and running it through the thing being measured would make
        # the measurement circular.
        result = source.execute(sql)
        return [list(row) for row in result.rows.itertuples(index=False)]

    return ask, execute
