"""The shape of a single conversational turn."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

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


@dataclass
class MaskedFrame:
    key: str
    frame: pd.DataFrame
    row_count: int
    redactions: int
    dropped_columns: tuple[str, ...] = ()

    def to_markdown(self, max_rows: int = 20) -> str:
        head = self.frame.head(max_rows)
        table = head.to_markdown(index=False)
        if self.row_count > max_rows:
            table += f"\n\n_({self.row_count - max_rows} more rows not shown)_"
        return table


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
