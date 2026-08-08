"""`retail-agent eval` — does the agent get the numbers right?

Every other test in this project asserts a path. This asserts an answer, by
running the agent against the warehouse and comparing what it computed to what a
hand-written reference query computes at the same moment.

Exit code is the point: 0 ships, 1 does not. A PII leak fails the run outright
however good the accuracy is.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from rich.console import Console

from retail_agent.config import get_settings
from retail_agent.evals.cases import EVAL_CASES
from retail_agent.evals.gate import DEFAULT_THRESHOLD
from retail_agent.evals.report import render_report, to_json
from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import CaseResult, EvalCase


def select_cases(
    cases: Sequence[EvalCase] = EVAL_CASES,
    *,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[EvalCase]:
    """Which cases to run, in corpus order.

    Corpus order rather than command-line order, so that two runs of the same
    set are comparable line by line.
    """
    chosen = list(cases)
    if ids:
        wanted = set(ids)
        # A typo would otherwise run zero cases and report a clean pass.
        unknown = wanted - {case.id for case in chosen}
        if unknown:
            raise ValueError(f"unknown case id(s): {', '.join(sorted(unknown))}")
        chosen = [case for case in chosen if case.id in wanted]
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def read_baseline(path: str | None) -> float | None:
    """The previous run's accuracy, or None.

    Never fatal. The first run has no baseline, and a corrupt one should cost
    the regression check rather than block the release outright — the threshold
    still applies either way.
    """
    if not path:
        return None
    try:
        with open(path) as handle:
            return float(json.load(handle)["accuracy"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_compare(args) -> int:
    """Put two `--json` reports side by side.

    Reads only what `eval` already writes, so the two arms never have to run in
    the same process — or on the same day.
    """
    console = Console()
    from retail_agent.evals.compare import compare_runs, load_run, render_comparison

    runs = []
    for path in (args.left, args.right):
        try:
            with open(path) as handle:
                runs.append(load_run(json.load(handle)))
        except (OSError, ValueError, KeyError, TypeError) as err:
            console.print(f"[red]Could not read {path}: {err}[/red]")
            return 1

    comparison = compare_runs(
        runs[0],
        runs[1],
        left_name=getattr(args, "left_name", None) or "before",
        right_name=getattr(args, "right_name", None) or "after",
    )
    # Plain print rather than rich markup: SQL in the disagreement list contains
    # brackets that rich would read as style tags and swallow.
    print(render_comparison(comparison))
    return 0


def run_evals(args) -> int:
    console = Console()
    settings = get_settings()

    try:
        cases = select_cases(
            ids=getattr(args, "case", None), limit=getattr(args, "limit", None)
        )
    except ValueError as err:
        console.print(f"[red]{err}[/red]")
        return 1

    # Imported here, not at module scope: this is what costs a second of
    # BigQuery and langchain imports, and `--help` should not pay for it.
    from retail_agent.agent.seams import build_seams
    from retail_agent.evals.runner import run_suite

    try:
        ask, execute = build_seams(settings)
    except Exception as err:
        console.print(f"[red]Could not reach the agent or the warehouse: {err}[/red]")
        return 1

    console.print(f"Running {len(cases)} cases against live BigQuery…\n")

    def show(result: CaseResult) -> None:
        mark = {Outcome.PASS: "[green]ok[/green]", Outcome.FAIL: "[red]WRONG[/red]"}.get(
            result.outcome, "[yellow]error[/yellow]"
        )
        detail = f" — {result.detail}" if result.outcome is not Outcome.PASS else ""
        leak = " [red]PII[/red]" if result.pii_leaked else ""
        console.print(f"  {result.case_id:32} {mark}{leak}{detail}")

    gate = run_suite(
        cases,
        ask=ask,
        execute=execute,
        threshold=(
            args.threshold if getattr(args, "threshold", None) is not None
            else DEFAULT_THRESHOLD
        ),
        baseline=read_baseline(getattr(args, "baseline", None)),
        on_case=show,
    )

    console.print()
    console.print(render_report(gate))

    path = getattr(args, "json_path", None)
    if path:
        with open(path, "w") as handle:
            handle.write(to_json(gate))
        console.print(f"\nWrote {path}")

    return 0 if gate.passed else 1
