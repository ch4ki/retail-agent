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
    # The result hit the display cap, so these rows are a sample.
    truncated: bool = False
    # How many rows matched in total — exact, even when only some were fetched.
    row_count: int = 0
    # What the turn cost. Filled in by the seam, which owns the callback
    # handler; the runner only carries it through. Counting inside the runner
    # would mean one implementation per arm, and two implementations of the
    # same measurement eventually disagree.
    tokens_in: int = 0
    tokens_out: int = 0
    # Round trips: node executions on the graph arm, tool calls on the ReAct
    # arm. Not the same unit, and the report says so — it is here to catch an
    # arm that wins on accuracy by doing far more work.
    calls: int = 0


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
            **_cost(answer),
        )

    if not reference:
        return CaseResult(
            case_id=case.id,
            outcome=Outcome.ERROR,
            detail="reference query returned no rows",
            answer=answer.text,
            sql=answer.sql,
            seconds=time.monotonic() - started,
            **_cost(answer),
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
    elif answer.truncated and not case.ranked:
        # The rows are a sample, so the first one is not the answer — scoring it
        # would report "expected 5823, got 1", reading as a confidently wrong
        # agent when the truth is the opposite.
        #
        # But the count of matching rows is exact, and for a counting question
        # it IS the answer: asked how many customers are loyal, the agent
        # returned one row per loyal customer. Comparing that count to the
        # reference is not prose-scoring — it is a number the warehouse
        # reported, and a query whose row count is wrong still fails.
        counted = compare(
            actual=answer.row_count, expected=expected, tolerance=case.tolerance
        )
        result = (
            (Outcome.PASS, "")
            if counted.outcome is Outcome.PASS
            else (
                Outcome.ERROR,
                f"rows are a sample ({len(answer.rows)} of {answer.row_count}); "
                "no total available",
            )
        )
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
        **_cost(answer),
    )


def _cost(answer: AgentAnswer) -> dict:
    """What the turn spent, for whichever `CaseResult` is being built.

    A helper rather than three copies: a return path that forgot it would report
    a free answer, and the cheapest arm would be whichever one had the bug.
    """
    return {
        "tokens_in": answer.tokens_in,
        "tokens_out": answer.tokens_out,
        "calls": answer.calls,
    }


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
