"""The shape of a single conversational turn."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

Intent = Literal["chat", "schema", "analyze"]
Status = Literal["ok", "degraded", "refused"]


@dataclass
class AnalysisStep:
    id: str
    question: str
    sql: str | None = None
    result_key: str | None = None


@dataclass
class SqlAttempt:
    step_id: str
    sql: str
    violations: tuple[str, ...] = ()
    error: str | None = None
    row_count: int | None = None
    bytes_billed: int | None = None

    @property
    def failed(self) -> bool:
        return bool(self.violations) or self.error is not None


# Checkpointed state must be serialisable, so results are stored as plain
# values rather than a DataFrame. Only a slice is kept: prompts show at most 20
# rows, and a checkpoint per turn should not carry the whole result set.
MAX_STORED_ROWS = 100


@dataclass
class MaskedFrame:
    key: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    redactions: int
    dropped_columns: tuple[str, ...] = ()

    @classmethod
    def from_dataframe(
        cls,
        key: str,
        frame: pd.DataFrame,
        *,
        row_count: int,
        redactions: int,
        dropped_columns: tuple[str, ...] = (),
    ) -> "MaskedFrame":
        head = frame.head(MAX_STORED_ROWS)
        return cls(
            key=key,
            columns=tuple(str(column) for column in frame.columns),
            rows=tuple(
                tuple(_cell(value) for value in row)
                for row in head.itertuples(index=False, name=None)
            ),
            row_count=row_count,
            redactions=redactions,
            dropped_columns=tuple(dropped_columns),
        )

    def column(self, name: str) -> list[Any]:
        index = self.columns.index(name)
        return [row[index] for row in self.rows]

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
    redactions: int
    answer: str
    status: Status


def new_turn_state(
    *, user_id: str, session_id: str, question: str, repair_budget: int
) -> TurnState:
    return TurnState(
        user_id=user_id,
        session_id=session_id,
        turn_id=uuid.uuid4().hex[:12],
        messages=[HumanMessage(content=question)],
        plan=[],
        step_index=0,
        sql_attempts=[],
        frames={},
        repair_budget=repair_budget,
        redactions=0,
        status="ok",
    )
