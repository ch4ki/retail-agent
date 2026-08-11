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

from retail_agent.agent.deps import AgentDeps, TurnContext
from retail_agent.agent.state import intent_from_events
from retail_agent.agent.subagents import final_text
from retail_agent.agent.supervisor import build_agent
from retail_agent.evals.runner import AgentAnswer
from retail_agent.evals.usage import UsageCollector


def answer_from_state(
    text: str,
    state: dict,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    report_bodies: Sequence[str] = (),
) -> AgentAnswer:
    """Reduce a finished turn's checkpointed `TurnState` to what an eval can score.

    The number comes from the recorded frame, never from `text`: parsing prose
    would measure how the model phrased itself, which legitimately varies
    between model versions, rather than what the query returned.

    `report_bodies` is supplied by the caller rather than read off `state`:
    a report's body is never written into state (`reports.py`'s own comment —
    "the report store's copy is the one that gets read"), so a caller wanting
    to score the content, not just the id, has to fetch it from the report
    store itself and pass it through.
    """
    frame = state.get("frame")
    events = state.get("events") or []

    return AgentAnswer(
        text=text,
        rows=[list(row) for row in frame["rows"]] if frame else [],
        columns=tuple(frame["columns"]) if frame else (),
        sql=state.get("executed_sql", ""),
        intent=intent_from_events(events),
        trios=tuple(state.get("trio_ids") or []),
        truncated=bool(frame["truncated"]) if frame else False,
        row_count=int(frame["row_count"]) if frame else 0,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        calls=state.get("calls", 0),
        # In order, not as a set: the suite scores whether the agent asked what
        # a term meant *before* it spent a query, and the sequence is the only
        # place that shows.
        tools=tuple(event["name"] for event in events),
        report_ids=tuple(
            report["report_id"] for report in (state.get("reports_written") or [])
        ),
        report_bodies=tuple(report_bodies),
    )


def ask_once(
    deps: AgentDeps,
    question: str,
    *,
    user: str = "eval",
    usage: UsageCollector | None = None,
) -> AgentAnswer:
    """One turn, reduced to a scorable answer.

    A fresh thread per call. Sharing one would let case 3's rows or case 3's
    conversation reach case 4, and the suite would be measuring memory rather
    than analysis.
    """
    session = f"eval-{uuid.uuid4().hex[:8]}"
    turn_id = uuid.uuid4().hex[:12]
    if usage is not None:
        usage.reset()

    agent = build_agent(deps, checkpointer=MemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": session}}
    if usage is not None:
        config["callbacks"] = [usage]

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        context=TurnContext(user_id=user, session_id=session, turn_id=turn_id),
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

    # Bodies fetched from the report store by id: `answer_from_state` cannot
    # read them off `state`, which never carries one — see its own docstring.
    report_bodies = tuple(
        report.body
        for entry in (result.get("reports_written") or [])
        if (report := deps.reports.get(owner_id=user, report_id=entry["report_id"]))
        is not None
    )

    return answer_from_state(
        final_text(result),
        result,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
        report_bodies=report_bodies,
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
