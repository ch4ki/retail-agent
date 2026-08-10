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
    # Tool names in the order they ran. Carried so the suite can score *how*
    # an answer was reached and not only whether it was right — specifically
    # whether the agent asked what a term meant before querying against a guess.
    tools: tuple[str, ...] = field(default_factory=tuple)
    # Reports this turn produced. Carried because the supervisor's answer is a
    # covering line now — a case that scored only `text` would pass on a report
    # that was never written. Bodies as well as ids, so a case can score the
    # content without a store round trip.
    report_ids: tuple[str, ...] = field(default_factory=tuple)
    report_bodies: tuple[str, ...] = field(default_factory=tuple)


Ask = Callable[[str], AgentAnswer]
Execute = Callable[[str], Sequence[Sequence[Any]]]

# The tool the agent calls when it does not understand a word, and the tool
# that spends money. The order between them is the whole measurement.
ASK_TOOL = "ask_for_definitions"
QUERY_TOOL = "run_sql"


def asked_before_querying(tools: Sequence[str]) -> bool:
    """Did the agent ask what a term meant before it spent a query?

    The ordering, not the call. An agent that queried against an invented
    definition and asked afterwards has already paid for the wrong number, and
    counting that as asking would report the bug as the fix.

    Asking and never querying counts: the executive cancelled, or handed the
    decision back and the turn ended some other way. The agent did its part.
    """
    if ASK_TOOL not in tools:
        return False
    if QUERY_TOOL not in tools:
        return True
    return tools.index(ASK_TOOL) < tools.index(QUERY_TOOL)


def ask_rate(results: Sequence[CaseResult]) -> float | None:
    """How often the agent asked, over the cases that had something to ask about.

    `None` when no case did. This is the number that replaced a guarantee: the
    old detector could not fail to fire, and a tool can simply not be called.
    Nothing enforces it any more, so it is measured instead — a rate that slips
    is the signal to put a deterministic backstop back in front of the analyst.
    """
    relevant = [r for r in results if r.asked_first is not None]
    if not relevant:
        return None
    return round(sum(1 for r in relevant if r.asked_first) / len(relevant), 4)


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
        **_route(case, answer),
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
        **_route(case, answer),
        )

    expected = _extract(reference, ranked=case.ranked)
    actual = _extract(
        answer.rows,
        ranked=case.ranked,
        columns=answer.columns,
        answer_column=case.answer_column,
    )

    if isinstance(actual, _NoAnswerColumn):
        result = Outcome.ERROR, actual.detail
    elif actual is None:
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
        **_route(case, answer),
    )


def _route(case: EvalCase, answer: AgentAnswer) -> dict:
    """Whether the agent asked before it queried, for the cases where it should.

    Separate from `_cost` because it needs the case: only a case naming
    `required_definitions` had anything to ask about, and scoring the rest would
    dilute the rate towards 100% with turns that were right to stay quiet.
    """
    if not case.required_definitions:
        return {"asked_first": None}
    return {"asked_first": asked_before_querying(answer.tools)}


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


@dataclass(frozen=True)
class _NoAnswerColumn:
    """A scalar case whose answer could not be picked without guessing.

    Guessing column 0 out of several is how user_id once scored as an order
    count — silently, with the miss reading as a confidently wrong agent.
    This scores as a visible extraction error naming both sides instead.
    """

    detail: str


def _extract(
    rows: Sequence[Sequence[Any]],
    *,
    ranked: bool,
    columns: Sequence[str] = (),
    answer_column: str = "",
) -> Any:
    """The comparable value: a column for a ranked case, one cell otherwise.

    `columns` is only ever passed for the agent's side. The reference side is
    the case author's own SQL, where the first column is the contract.
    """
    if not rows:
        return None
    if ranked:
        return [row[0] if row else None for row in rows]

    first = rows[0]
    if not first:
        return None

    # A scalar question answered with several columns: take the one the case
    # named, so extra context the agent chose to return does not score as a
    # wrong answer. A single column is unambiguous whatever it is called.
    if columns and len(first) > 1:
        wanted = answer_column.strip().lower()
        for index, name in enumerate(columns):
            if str(name).strip().lower() == wanted and index < len(first):
                return first[index]
        offered = ", ".join(str(name) for name in columns)
        if wanted:
            return _NoAnswerColumn(
                f"the case names answer_column {answer_column!r} but the agent "
                f"returned columns ({offered}); scoring any of them would be a "
                "guess"
            )
        return _NoAnswerColumn(
            f"the agent returned several columns ({offered}) and the case "
            "names no answer_column; add one rather than let the harness "
            "guess"
        )

    return first[0]
