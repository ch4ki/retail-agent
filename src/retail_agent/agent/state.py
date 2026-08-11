"""What a turn did, as graph state.

This replaces `TurnCapture`, which every tool closed over and mutated in place.
Accumulation bound at build time is why the agent had to be rebuilt for each
turn, and why the record of a turn was not checkpointed: a conversation resumed
from a checkpoint carried its messages but not what its tools had done.

Everything here is a plain primitive. LangGraph round-trips custom dataclasses
today but warns that deserializing unregistered types "will be blocked in a
future version", and state is checkpointed on every super-step — so a
`MaskedFrame` in state is both a deprecation and a size problem. `events` is a
list of dicts rather than tuples because the serializer returns a list for a
tuple, and an annotation the checkpointer contradicts is worse than a clumsy one.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated

from langchain.agents.middleware import AgentState


class TurnState(AgentState):
    """The turn's record, alongside the messages.

    The `operator.add` fields accumulate across tool calls; without a reducer
    the second `run_sql` in a turn would replace the first rather than add to
    it, and the turn would report one query.
    """

    attempts: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]
    trio_ids: Annotated[list[str], operator.add]
    assumed_terms: Annotated[list[str], operator.add]
    preference_changes: Annotated[list[dict], operator.add]
    reports_written: Annotated[list[dict], operator.add]
    redactions: Annotated[int, operator.add]
    calls: Annotated[int, operator.add]
    context_tokens: int
    frame: dict | None
    executed_sql: str
    status: str


def step_event(name: str, started: float, detail: str = "") -> dict:
    """One tool call, timed. What `capture.step` used to append.

    `started` is a `time.perf_counter()` reading taken before the work.
    """
    return {
        "name": name,
        "ms": int((time.perf_counter() - started) * 1000),
        "detail": detail,
    }


def frame_to_state(frame) -> dict:
    """A `MaskedFrame` as plain values.

    Rows become lists rather than tuples so what a consumer reads after a
    resume is what it read before one.
    """
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": [list(row) for row in frame.rows],
        "row_count": frame.row_count,
        "redactions": frame.redactions,
        "dropped_columns": list(frame.dropped_columns),
        "truncated": frame.truncated,
    }


def attempt_record(
    *,
    sql: str,
    executed_sql: str | None = None,
    violations: list[str] | None = None,
    error: str | None = None,
    row_count: int | None = None,
    bytes_billed: int | None = None,
    index: int = 0,
) -> dict:
    """One query attempt. The same keys `TurnCapture.record_attempt` built.

    `index` is the caller's count of attempts so far, so `step_id` still reads
    `q1`, `q2`, … — the state's reducer cannot know the running total.
    """
    return {
        "step_id": f"q{index + 1}",
        "sql": sql,
        "executed_sql": executed_sql or None,
        "violations": list(violations or []),
        "error": error,
        "row_count": row_count,
        "bytes_billed": bytes_billed,
    }
