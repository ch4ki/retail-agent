"""Rendering a run.

Someone reads this when the gate blocks a release, usually in a hurry. It states
the verdict, then the failures with both numbers, and says nothing else — a
report that lists forty passing cases buries the three that matter.

Wrong answers and errors are counted separately throughout. An agent that says
"I could not work that out" needs a different fix from one that confidently
returns 1254 when the answer is 5746, and a single "60% correct" hides which of
the two is happening.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import Gate


def render_report(gate: Gate) -> str:
    verdict = "PASS" if gate.passed else "FAIL"
    wrong = [r for r in gate.results if r.outcome is Outcome.FAIL]
    errored = [r for r in gate.results if r.outcome is Outcome.ERROR]
    leaked = [r for r in gate.results if r.pii_leaked]

    lines = [
        f"{verdict} — {gate.reason}",
        "",
        f"{len(gate.results)} cases: "
        f"{len(gate.results) - len(wrong) - len(errored)} correct, "
        f"{len(wrong)} wrong, {len(errored)} error",
    ]

    if leaked:
        lines += [
            "",
            f"PII LEAKED in {len(leaked)} case(s): {', '.join(r.case_id for r in leaked)}",
        ]

    if wrong or errored:
        lines += ["", "Failures", "--------"]
    for result in wrong:
        lines.append(
            f"  {result.case_id}: expected {_plain(result.expected)}, "
            f"got {_plain(result.actual)}"
        )
        if result.detail:
            lines.append(f"      {result.detail}")
        if result.used_trios:
            # Distinguishes "never retrieved a definition" from "had the
            # definition and still got it wrong" — different bugs entirely.
            lines.append(f"      trios: {', '.join(result.used_trios)}")
        if result.sql:
            lines.append(f"      sql: {_one_line(result.sql)}")
    for result in errored:
        lines.append(f"  {result.case_id}: {result.detail}")

    return "\n".join(lines)


def to_json(gate: Gate) -> str:
    """The machine-readable form, which the next run reads back as its baseline."""
    return json.dumps(
        {
            "passed": gate.passed,
            "accuracy": gate.accuracy,
            "reason": gate.reason,
            "results": [
                {**asdict(r), "outcome": r.outcome.value} for r in gate.results
            ],
        },
        indent=2,
        default=str,
    )


def _plain(value) -> str:
    """Readable, not repr'd.

    BigQuery hands back numpy scalars, and "expected np.int64(5746)" in a
    failure report reads like a bug in the eval rather than in the agent.
    """
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(_plain(v) for v in value)}]"
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar → the Python number behind it
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _one_line(sql: str, limit: int = 120) -> str:
    flat = " ".join(sql.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"
