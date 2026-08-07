"""The `retail-agent eval` command.

Case selection and baseline reading are pure and tested here. The part that
needs BigQuery and an LLM key is one function call away, behind `build_seams`.
"""

from __future__ import annotations

import json

import pytest

from retail_agent.cli.evals import read_baseline, select_cases
from retail_agent.evals.types import EvalCase

CASES = (
    EvalCase(id="a", question="q", reference_sql="s"),
    EvalCase(id="b", question="q", reference_sql="s"),
    EvalCase(id="c", question="q", reference_sql="s"),
)


def test_no_selection_runs_everything():
    assert select_cases(CASES) == list(CASES)


def test_named_cases_are_selected_in_corpus_order():
    """Order comes from the corpus, not from the command line, so two runs of
    the same set are comparable."""
    assert [c.id for c in select_cases(CASES, ids=["c", "a"])] == ["a", "c"]


def test_an_unknown_case_id_is_an_error_not_a_silent_empty_run():
    """A typo would otherwise run zero cases and report a clean pass."""
    with pytest.raises(ValueError, match="nope"):
        select_cases(CASES, ids=["nope"])


def test_limit_takes_the_first_n():
    assert [c.id for c in select_cases(CASES, limit=2)] == ["a", "b"]


def test_reading_a_baseline_returns_its_accuracy(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"accuracy": 0.92, "passed": True, "results": []}))

    assert read_baseline(str(path)) == 0.92


def test_a_missing_baseline_is_not_fatal(tmp_path):
    """The first run has no baseline, and CI must not fail because of that."""
    assert read_baseline(str(tmp_path / "absent.json")) is None


def test_an_unreadable_baseline_is_not_fatal(tmp_path):
    """A corrupt baseline should not block a release; the threshold still
    applies, and the regression check is the part that degrades."""
    path = tmp_path / "broken.json"
    path.write_text("{not json")

    assert read_baseline(str(path)) is None


def test_no_baseline_path_means_no_baseline():
    assert read_baseline(None) is None


# --- choosing which agent answers ---


def test_the_graph_is_the_arm_that_runs_by_default():
    """Adding a second arm must not quietly change what `retail-agent eval`
    has always meant."""
    from retail_agent.cli.evals import seams_builder
    from retail_agent.evals.harness import build_seams

    assert seams_builder(None) is build_seams
    assert seams_builder("graph") is build_seams


def test_the_react_arm_is_selectable():
    from retail_agent.baseline.seams import build_react_seams
    from retail_agent.cli.evals import seams_builder

    assert seams_builder("react") is build_react_seams


def test_an_unknown_arm_is_refused_rather_than_silently_defaulting():
    """A typo that fell through to the graph would produce a report labelled
    react containing the graph's numbers."""
    import pytest

    from retail_agent.cli.evals import seams_builder

    with pytest.raises(ValueError, match="grpah"):
        seams_builder("grpah")


# --- the comparison command ---


def test_comparing_two_reports_prints_both_arms(tmp_path, capsys):
    from retail_agent.cli.evals import run_compare
    from retail_agent.evals.report import to_json
    from retail_agent.evals.scoring import Outcome
    from retail_agent.evals.types import CaseResult, Gate

    def report(outcome):
        return to_json(
            Gate(
                passed=True,
                accuracy=1.0 if outcome is Outcome.PASS else 0.0,
                reason="",
                results=(CaseResult(case_id="loyal-count", outcome=outcome),),
            )
        )

    left = tmp_path / "graph.json"
    right = tmp_path / "react.json"
    left.write_text(report(Outcome.PASS))
    right.write_text(report(Outcome.FAIL))

    code = run_compare(_Args(left=str(left), right=str(right)))

    printed = capsys.readouterr().out
    assert code == 0
    assert "graph" in printed
    assert "react" in printed
    assert "loyal-count" in printed


def test_comparing_against_a_missing_report_fails_cleanly(tmp_path, capsys):
    from retail_agent.cli.evals import run_compare

    code = run_compare(_Args(left=str(tmp_path / "nope.json"), right=str(tmp_path)))

    assert code == 1
    assert "nope.json" in capsys.readouterr().out


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
