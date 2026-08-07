"""Two runs, side by side.

The headline numbers are an index into the disagreements, not the finding. An
accuracy delta of four cases says nothing about *which* four, and the whole
reason to build a second arm was to learn where control flow changed the answer
— on the definition cases, on the empty results, or on multi-step questions.

Deliberately not a gate. `evaluate_gate` decides ship or do-not-ship for one
agent; this decides nothing and reports everything.
"""

from __future__ import annotations

from dataclasses import dataclass

from retail_agent.evals.report import _one_line, _plain
from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import CaseResult, Gate


@dataclass(frozen=True)
class ArmSummary:
    name: str
    accuracy: float
    cases: int
    mean_tokens_in: float
    mean_tokens_out: float
    mean_calls: float
    mean_seconds: float
    leaks: tuple[str, ...]


@dataclass(frozen=True)
class Disagreement:
    case_id: str
    left: CaseResult
    right: CaseResult


@dataclass(frozen=True)
class Comparison:
    left: ArmSummary
    right: ArmSummary
    disagreements: tuple[Disagreement, ...]
    # Named rather than counted: a run that died partway would otherwise
    # compare as a clean sweep of whatever it managed to finish.
    only_in_left: tuple[str, ...]
    only_in_right: tuple[str, ...]


def compare_runs(
    left: Gate, right: Gate, *, left_name: str = "graph", right_name: str = "react"
) -> Comparison:
    by_left = {r.case_id: r for r in left.results}
    by_right = {r.case_id: r for r in right.results}
    shared = [case_id for case_id in by_left if case_id in by_right]

    return Comparison(
        left=_summarise(left_name, left),
        right=_summarise(right_name, right),
        disagreements=tuple(
            Disagreement(case_id, by_left[case_id], by_right[case_id])
            for case_id in shared
            if by_left[case_id].outcome is not by_right[case_id].outcome
        ),
        only_in_left=tuple(c for c in by_left if c not in by_right),
        only_in_right=tuple(c for c in by_right if c not in by_left),
    )


def load_run(payload: dict) -> Gate:
    """Read back what `report.to_json` wrote.

    The same files `eval --json` already produces. A second serialisation format
    would be a second thing to keep in step with `CaseResult`.
    """
    results = tuple(
        CaseResult(
            **{
                **{k: v for k, v in row.items() if k != "outcome"},
                "outcome": Outcome(row["outcome"]),
                "used_trios": tuple(row.get("used_trios") or ()),
            }
        )
        for row in payload.get("results", [])
    )
    return Gate(
        passed=bool(payload.get("passed")),
        accuracy=float(payload.get("accuracy") or 0.0),
        reason=str(payload.get("reason") or ""),
        results=results,
    )


def render_comparison(
    comparison: Comparison, *, left_name: str = "", right_name: str = ""
) -> str:
    left = comparison.left
    right = comparison.right
    # The names on the summaries win; the keywords exist so a caller can label
    # a comparison built from bare gates.
    left_label = left.name or left_name or "left"
    right_label = right.name or right_name or "right"

    lines = [
        f"{'':22} {left_label:>14} {right_label:>14}",
        f"{'accuracy':22} {left.accuracy:>13.0%} {right.accuracy:>13.0%}",
        f"{'cases':22} {left.cases:>14} {right.cases:>14}",
        f"{'mean tokens in':22} {left.mean_tokens_in:>14,.0f} "
        f"{right.mean_tokens_in:>14,.0f}",
        f"{'mean tokens out':22} {left.mean_tokens_out:>14,.0f} "
        f"{right.mean_tokens_out:>14,.0f}",
        f"{'mean calls':22} {left.mean_calls:>14.1f} {right.mean_calls:>14.1f}",
        f"{'mean seconds':22} {left.mean_seconds:>14.1f} {right.mean_seconds:>14.1f}",
        # Kept beside accuracy rather than folded into it. A leak is a blocking
        # failure however good the numbers are, and averaging the two together
        # would let one hide the other.
        f"{'PII leaks':22} {len(left.leaks):>14} {len(right.leaks):>14}",
    ]

    if comparison.only_in_left or comparison.only_in_right:
        lines.append("")
        lines.append("Cases missing from one run:")
        for case_id in comparison.only_in_left:
            lines.append(f"  {case_id}: absent from {right_label}")
        for case_id in comparison.only_in_right:
            lines.append(f"  {case_id}: absent from {left_label}")

    lines.append("")
    if not comparison.disagreements:
        lines.append("The two arms agree on every shared case.")
        return "\n".join(lines)

    lines.append(f"Disagreements ({len(comparison.disagreements)}):")
    for item in comparison.disagreements:
        lines.append("")
        lines.append(f"  {item.case_id}")
        for label, side in ((left_label, item.left), (right_label, item.right)):
            lines.append(
                f"    {label:>8}: {side.outcome.value:<7} "
                f"got {_plain(side.actual)}, expected {_plain(side.expected)}"
            )
            if side.detail:
                lines.append(f"    {'':>8}  {side.detail}")
            if side.sql:
                lines.append(f"    {'':>8}  {_one_line(side.sql)}")

    return "\n".join(lines)


def _summarise(name: str, gate: Gate) -> ArmSummary:
    results = gate.results
    return ArmSummary(
        name=name,
        accuracy=gate.accuracy,
        cases=len(results),
        mean_tokens_in=_mean(r.tokens_in for r in results),
        mean_tokens_out=_mean(r.tokens_out for r in results),
        mean_calls=_mean(r.calls for r in results),
        mean_seconds=_mean(r.seconds for r in results),
        leaks=tuple(r.case_id for r in results if r.pii_leaked),
    )


def _mean(values) -> float:
    """Zero for an empty run rather than a ZeroDivisionError.

    A comparison of two runs that produced nothing should print zeroes and let
    the reader see there were no cases, not crash on the way to saying so.
    """
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
