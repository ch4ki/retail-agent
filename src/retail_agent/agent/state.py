"""The shape of a single conversational turn."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

Intent = Literal["chat", "schema", "analyze", "report_op"]
Status = Literal["ok", "degraded", "refused"]


@dataclass
class AnalysisStep:
    id: str
    question: str
    sql: str | None = None


@dataclass
class SqlAttempt:
    """One try at answering a step: what the model wrote, and what became of it.

    `draft_sql` opens the record with the guard's verdict; `execute` completes
    the same record with the outcome. One row per attempt rather than one per
    node visit, so `len(sql_attempts)` is the number of tries — the count the
    CLI shows the user and the denominator of the self-correction rate.
    """

    step_id: str
    sql: str  # as the model wrote it
    violations: tuple[str, ...] = ()
    executed_sql: str | None = None  # after the guard qualified and limited it
    error: str | None = None
    row_count: int | None = None
    bytes_billed: int | None = None
    # A diagnosis is not a try. It is recorded as a failed attempt so that
    # `draft_sql` reuses its repair prompt without knowing diagnosis exists,
    # but it never ran, so it must not be counted as an attempt the user made.
    is_diagnosis: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.violations) or self.error is not None


@dataclass
class TurnEvent:
    """One node execution: what ran, how long it took, and what came of it.

    Recorded by the graph rather than by the nodes, so a new node is traced the
    moment it is registered and cannot forget to report itself.
    """

    node: str
    duration_ms: int
    detail: str = ""


@dataclass
class PendingAction:
    """A destructive operation resolved against the store, awaiting a typed
    confirmation.

    Staged by `report_ops`, read by `apply_delete` on the far side of the
    breakpoint. It lives in graph state rather than in a transient payload so
    that `get_state()` shows exactly what is pending.
    """

    action_id: str
    report_ids: tuple[str, ...]
    titles: tuple[str, ...]
    token: str  # what the user must type back, verbatim


# Checkpointed state must be serialisable, so results are stored as plain
# values rather than a DataFrame. Only a slice is kept: prompts show at most 20
# rows, and a checkpoint per turn should not carry the whole result set.
MAX_STORED_ROWS = 100


@dataclass
class MaskedFrame:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    redactions: int
    dropped_columns: tuple[str, ...] = ()
    # The guard caps every query so a result stays printable and affordable,
    # and that cap is invisible in the result: 500 rows returned looks the same
    # whether there were 500 or 5,823. Measured on the real warehouse, the query
    # written for "how many loyal customers" matched 5,823 and the agent saw
    # 500. This is what stops a sample being read as the whole answer.
    truncated: bool = False

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        row_count: int,
        redactions: int,
        dropped_columns: tuple[str, ...] = (),
        truncated: bool = False,
    ) -> "MaskedFrame":
        head = frame.head(MAX_STORED_ROWS)
        return cls(
            columns=tuple(str(column) for column in frame.columns),
            rows=tuple(
                tuple(_cell(value) for value in row)
                for row in head.itertuples(index=False, name=None)
            ),
            row_count=row_count,
            redactions=redactions,
            dropped_columns=tuple(dropped_columns),
            truncated=truncated,
        )

    def to_markdown(self, max_rows: int = 20) -> str:
        if not self.columns:
            return "_(no rows)_"

        shown = self.rows[:max_rows]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join("---" for _ in self.columns) + " |",
            *("| " + " | ".join(_fmt(v) for v in row) + " |" for row in shown),
        ]
        table = "\n".join(lines)

        hidden = self.row_count - len(shown)
        if hidden > 0:
            table += f"\n\n_({hidden} more rows not shown)_"
        return table


def _cell(value: Any) -> Any:
    """Coerce a pandas/numpy value into something a checkpointer can store."""
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # arrays and other non-scalars land here

    if isinstance(value, (bool, int, float, str)):
        return value

    unwrap = getattr(value, "item", None)  # numpy scalars
    if callable(unwrap):
        try:
            return _cell(unwrap())
        except (AttributeError, TypeError, ValueError):
            pass

    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


class TurnState(TypedDict, total=False):
    user_id: str
    session_id: str
    turn_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent
    plan: list[AnalysisStep]
    step_index: int
    sql_attempts: list[SqlAttempt]
    frames: dict[str, MaskedFrame]
    repair_budget: int
    diagnose_budget: int
    redactions: int
    answer: str
    status: Status
    pending_action: PendingAction | None
    events: list[TurnEvent]
    confirmation: str
    trio_ids: list[str]
    assumed_terms: list[str]
    personal_terms: list[str]
    pending_term: str
    definition_reply: str
    declined_terms: list[str]


def fresh_scratch(*, repair_budget: int, diagnose_budget: int = 1) -> dict:
    """Everything one turn accumulates. Nothing here may survive into the next.

    Applied by `start_turn`, the graph's first node, so it holds on every entry
    point: the CLI, LangGraph Studio, and a server invoking the compiled graph
    with nothing but `messages` on a thread that already has state.

    A field left out of this dict is inherited by the following turn. That has
    cost us a repair budget that was already spent (so the turn degraded having
    never retried) and a result frame that let a failed turn answer with the
    previous turn's numbers. `test_every_turn_state_field_is_classified` fails
    if a new `TurnState` field is not listed here or declared durable.
    """
    return {
        "turn_id": uuid.uuid4().hex[:12],
        "intent": "analyze",
        "plan": [],
        "step_index": 0,
        "sql_attempts": [],
        "frames": {},
        "redactions": 0,
        "answer": "",
        "status": "ok",
        "repair_budget": repair_budget,
        "diagnose_budget": diagnose_budget,
        "pending_action": None,
        "events": [],
        "confirmation": "",
        "trio_ids": [],
        "assumed_terms": [],
        "personal_terms": [],
        "pending_term": "",
        "definition_reply": "",
        "declined_terms": [],
    }


def new_turn_state(*, user_id: str, session_id: str, question: str) -> TurnState:
    """Seed a CLI turn. Scratch is the graph's job — see `fresh_scratch`."""
    return TurnState(
        user_id=user_id,
        session_id=session_id,
        messages=[HumanMessage(content=question)],
    )
