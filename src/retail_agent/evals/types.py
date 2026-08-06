"""The shapes an eval run passes around.

Separate from `cases` and `gate` so that scoring, gating and reporting can be
tested without importing the agent, BigQuery or an LLM — the harness has to be
verifiable on a machine with no credentials, or it will only ever be trusted by
whoever can run it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retail_agent.evals.scoring import Outcome


@dataclass(frozen=True)
class EvalCase:
    """One question, and how to know whether the answer was right.

    Ground truth is `reference_sql` rather than a frozen number. theLook is
    appended to continuously — its most recent order is dated today — so a
    literal `expected_value` starts rotting the day it is written, and the suite
    would fail for reasons that have nothing to do with the agent. The reference
    query is re-executed on every run, so the truth moves with the data.
    """

    id: str
    question: str
    reference_sql: str
    # Business terms the answer depends on. If the corpus does not define them
    # the agent is guessing, and the case is measuring the guess.
    required_definitions: tuple[str, ...] = ()
    expected_intent: str = ""
    tolerance: float = 1e-6
    # Ranked answers compare row by row; scalars take the single cell.
    ranked: bool = False
    # Which of the agent's columns holds the answer, when it returns more than
    # one. Live: asked which month of 2023 was busiest, the agent returned
    # (year, month, total_orders) with the right month in it, and taking column
    # 0 scored 2023 against an expected 12. A hint, not a contract — the agent
    # picks its own aliases, so a name that is not there falls back to the first
    # column.
    answer_column: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    outcome: Outcome
    pii_leaked: bool = False
    detail: str = ""
    actual: Any = None
    expected: Any = None
    answer: str = ""
    sql: str = ""
    intent: str = ""
    used_trios: tuple[str, ...] = ()
    seconds: float = 0.0


@dataclass(frozen=True)
class Gate:
    passed: bool
    accuracy: float
    reason: str
    results: tuple[CaseResult, ...] = field(default_factory=tuple)
