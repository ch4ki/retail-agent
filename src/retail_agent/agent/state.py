"""What a turn did, as graph state.

This replaces the closure every tool used to mutate in place. Accumulation
bound at build time is why the agent had to be rebuilt for each turn, and why
the record of a turn was not checkpointed: a conversation resumed from a
checkpoint carried its messages but not what its tools had done.

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


def _keep_last(existing, update):
    """Last write wins.

    `frame` and `executed_sql` are single-value fields, not accumulators —
    but a bare field with no reducer at all is `LastValue`, which raises
    `InvalidUpdateError` the moment two tool calls in the same super-step
    both write it. Gemini and OpenAI both emit parallel tool calls routinely
    (three `run_sql` calls in one assistant turn; a supervisor issuing two
    parallel `analyst` calls), so "only one tool ever runs per step" is not a
    safe assumption. An explicit reducer — even one that just keeps the
    newest value — turns that into an ordinary sequential apply instead of a
    crash.
    """
    return update


def _dedupe_ordered(existing, update):
    """Order-preserving union, the way `TurnCapture.record_definitions` and
    `record_assumptions` did it.

    `operator.add` appends, so the model looking the same term up twice would
    report it twice — and a forced disclosure that names an assumed term
    twice in one sentence reads as a mistake, not as thoroughness. A trace
    should report what was consulted, not how often.
    """
    result = list(existing)
    for item in update:
        if item not in result:
            result.append(item)
    return result


class TurnState(AgentState):
    """The turn's record, alongside the messages.

    The `operator.add` fields accumulate across tool calls; without a reducer
    the second `run_sql` in a turn would replace the first rather than add to
    it, and the turn would report one query. `trio_ids` and `assumed_terms`
    use `_dedupe_ordered` instead — see its docstring. `frame` and
    `executed_sql` use `_keep_last` — see its docstring.
    """

    attempts: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]
    trio_ids: Annotated[list[str], _dedupe_ordered]
    assumed_terms: Annotated[list[str], _dedupe_ordered]
    preference_changes: Annotated[list[dict], operator.add]
    reports_written: Annotated[list[dict], operator.add]
    redactions: Annotated[int, operator.add]
    calls: Annotated[int, operator.add]
    context_tokens: int
    frame: Annotated[dict | None, _keep_last]
    executed_sql: Annotated[str, _keep_last]
    status: str


def step_event(name: str, started: float, detail: str = "") -> dict:
    """One tool call, timed — filed by the call site, now that a turn's
    record is checkpointed state rather than a closure a step could wrap.

    `started` is a `time.perf_counter()` reading taken before the work.
    """
    return {
        "name": name,
        "ms": int((time.perf_counter() - started) * 1000),
        "detail": detail,
    }


def intent_from_events(events) -> str:
    """Which of four things the turn was, read off which tools ran.

    Derived rather than classified — the graph used to ask a model which of
    four things a turn was before doing any of them. Which tools actually ran
    is the same answer, arrived at after the fact and for free. `events` is
    `TurnState["events"]`, a list of `{"name": ..., "ms": ..., "detail": ...}`
    dicts.
    """
    ran = {event["name"] for event in events}
    if "analyst" in ran or "run_sql" in ran:
        return "analyze"
    if ran & {"report_writer", "ask_about_report", "list_reports", "delete_reports"}:
        return "report_op"
    if "describe_schema" in ran:
        return "schema"
    return "chat"


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
    """One query attempt, as a plain dict.

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
