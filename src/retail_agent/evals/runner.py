"""Running one case and scoring it.

Two seams — `ask` and `execute` — rather than a hard dependency on the graph and
BigQuery. That is what lets the harness be tested with no credentials, no key
and no warehouse: a suite only its author can run is a suite only its author
trusts. The live wiring that fills those seams in is in `harness.py`.

The number is read from the *result rows*, never from the narrative. Parsing
prose would measure how the model phrased itself, which legitimately varies
between model versions, instead of what the query returned.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from retail_agent.evals.gate import DEFAULT_THRESHOLD, evaluate_gate
from retail_agent.evals.scoring import Outcome, compare
from retail_agent.evals.types import CaseResult, EvalCase, Gate
from retail_agent.safety.egress import scan_text


@dataclass(frozen=True)
class AgentAnswer:
    """What one turn produced, reduced to what an eval can score."""

    text: str
    rows: Sequence[Sequence[Any]]
    columns: tuple[str, ...] = ()
    sql: str = ""
    intent: str = ""
    trios: tuple[str, ...] = field(default_factory=tuple)


Ask = Callable[[str], AgentAnswer]
Execute = Callable[[str], Sequence[Sequence[Any]]]


def run_case(case: EvalCase, *, ask: Ask, execute: Execute) -> CaseResult:
    """Ask the agent, run the reference query, and compare the two."""
    started = time.monotonic()

    try:
        answer = ask(case.question)
    except Exception as err:
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.ERROR,
            detail=f"the agent failed: {err}",
            seconds=time.monotonic() - started,
        )

    # A failure here is the eval suite's own fault, and reporting it as an agent
    # failure would send someone debugging the wrong thing.
    try:
        reference = execute(case.reference_sql)
    except Exception as err:
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.ERROR,
            detail=f"reference query failed: {err}",
            answer=answer.text,
            sql=answer.sql,
            seconds=time.monotonic() - started,
        )

    if not reference:
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.ERROR,
            detail="reference query returned no rows",
            answer=answer.text,
            sql=answer.sql,
            seconds=time.monotonic() - started,
        )

    expected = _extract(reference, ranked=case.ranked)
    actual = _extract(
        answer.rows,
        ranked=case.ranked,
        columns=answer.columns,
        answer_column=case.answer_column,
    )

    if actual is None:
        result = Outcome.ERROR, "the agent returned no rows to score"
    else:
        scored = compare(actual=actual, expected=expected, tolerance=case.tolerance)
        detail = scored.detail
        if scored.outcome is not Outcome.PASS and not case.ranked and len(answer.rows) > 1:
            # A scalar question answered with many rows is a wrong *shape*, not
            # an off-by-one. Seen live: `GROUP BY u.id` returned one row per
            # customer each containing 1, and the report said only "got 1".
            detail = f"{detail} (scalar question, but {len(answer.rows)} rows came back)"
        result = scored.outcome, detail

    # Scanned independently of correctness: a right answer that leaks an email
    # address is still a blocking problem, and folding the two together would
    # let one hide the other.
    leaked = bool(scan_text(answer.text).findings)

    return CaseResult(
        case_id=case.id,
        outcome=result[0],
        detail=result[1],
        pii_leaked=leaked,
        actual=actual,
        expected=expected,
        answer=answer.text,
        sql=answer.sql,
        intent=answer.intent,
        used_trios=tuple(answer.trios),
        seconds=time.monotonic() - started,
    )


def run_suite(
    cases: Sequence[EvalCase],
    *,
    ask: Ask,
    execute: Execute,
    threshold: float = DEFAULT_THRESHOLD,
    baseline: float | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
) -> Gate:
    """Run every case and gate the result.

    Sequential on purpose. Running forty LLM turns concurrently is the fastest
    way to hit a provider rate limit and produce a report full of errors that
    say nothing about the agent.
    """
    results = []
    for case in cases:
        result = run_case(case, ask=ask, execute=execute)
        results.append(result)
        if on_case is not None:
            # A live run is minutes of silence otherwise, with no way to tell a
            # slow case from a hung one.
            on_case(result)

    return evaluate_gate(results, threshold=threshold, baseline=baseline)


def _extract(
    rows: Sequence[Sequence[Any]],
    *,
    ranked: bool,
    columns: Sequence[str] = (),
    answer_column: str = "",
) -> Any:
    """The comparable value: a column for a ranked case, one cell otherwise."""
    if not rows:
        return None
    if ranked:
        return [row[0] if row else None for row in rows]

    first = rows[0]
    if not first:
        return None

    # A scalar question answered with several columns: take the one the case
    # named, so extra context the agent chose to return does not score as a
    # wrong answer.
    if answer_column and columns:
        wanted = answer_column.strip().lower()
        for index, name in enumerate(columns):
            if str(name).strip().lower() == wanted and index < len(first):
                return first[index]

    return first[0]
