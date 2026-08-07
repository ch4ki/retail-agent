"""Wiring the eval seams to the real agent and the real warehouse.

`run_case` deliberately knows nothing about LangGraph or BigQuery. This is where
that gets connected, and it is the only part of the eval suite that needs
credentials — which is why the extraction below is a pure function tested
separately rather than something only a live run can exercise.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from retail_agent.evals.runner import AgentAnswer


def answer_from_state(state: dict) -> AgentAnswer:
    """Reduce a finished turn to what an eval can score.

    The number comes from the result frame, never from the narrative: parsing
    prose would measure how the model phrased itself, which legitimately varies
    between model versions, rather than what the query returned.
    """
    frames = state.get("frames") or {}
    frame = _final_frame(state, frames)

    return AgentAnswer(
        text=state.get("answer") or "",
        rows=[list(row) for row in frame.rows] if frame else [],
        columns=tuple(frame.columns) if frame else (),
        sql=_executed_sql(state),
        intent=str(state.get("intent") or ""),
        trios=tuple(state.get("trio_ids") or ()),
        truncated=bool(getattr(frame, "truncated", False)) if frame else False,
    )


def _final_frame(state: dict, frames: dict) -> Any:
    """The last step's frame — an earlier one is working, not the answer."""
    plan = state.get("plan") or []
    for step in reversed(plan):
        step_id = getattr(step, "id", None) or (step.get("id") if isinstance(step, dict) else None)
        if step_id in frames:
            return frames[step_id]

    # A degraded turn can leave frames behind without a readable plan. Losing
    # the number here would report ERROR for a turn that did in fact answer.
    return list(frames.values())[-1] if frames else None


def _executed_sql(state: dict) -> str:
    """The query that ran, not the first draft: a repaired turn has several
    attempts and only the last one produced the number being scored."""
    for attempt in reversed(state.get("sql_attempts") or []):
        executed = getattr(attempt, "executed_sql", None)
        if executed:
            return executed
    return ""


def build_seams(settings, *, user: str = "eval"):
    """The `ask` and `execute` callables `run_case` needs, bound to real
    services. Imported lazily by the CLI so that `retail-agent eval --help`
    does not pay for BigQuery and langchain."""
    from retail_agent.agent.graph import build_graph, run_turn
    from retail_agent.bootstrap import build_deps
    from retail_agent.datasources.bigquery import BigQuerySource
    from retail_agent.llm.provider import build_llm

    from langgraph.checkpoint.memory import MemorySaver

    source = BigQuerySource(settings)
    deps = build_deps(settings, llm=build_llm(settings), source=source)
    # Required, not optional: the graph has static breakpoints, and LangGraph
    # refuses to run one without somewhere to checkpoint. In memory rather than
    # Postgres because an eval thread is thrown away at the end of the case —
    # persisting forty of them per run would be noise in the user's history.
    graph = build_graph(deps, checkpointer=MemorySaver())

    def ask(question: str) -> AgentAnswer:
        # A fresh thread per case. Sharing one would let the answer to case 3
        # sit in the history of case 4, and the suite would be measuring
        # conversation memory rather than analysis.
        session = f"eval-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": session}}
        state = run_turn(
            graph, user_id=user, session_id=session, question=question, config=config
        )

        # A case that pauses for confirmation or a definition never produces a
        # number. Recorded as an unanswered turn rather than hanging the run.
        if getattr(graph.get_state(config), "next", ()):
            return AgentAnswer(text="[paused awaiting input]", rows=[], columns=())

        return answer_from_state(state)

    def execute(sql: str) -> Sequence[Sequence[Any]]:
        # The reference query bypasses the guard and the agent entirely: it is
        # the ruler, and running it through the thing being measured would make
        # the measurement circular.
        result = source.execute(sql)
        return [list(row) for row in result.rows.itertuples(index=False)]

    return ask, execute
