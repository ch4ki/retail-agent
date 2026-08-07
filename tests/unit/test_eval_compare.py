"""Putting two runs side by side.

The headline numbers are an index; the disagreements are the product. A single
accuracy delta cannot say whether one arm lost on definitions, on empty results
or on plan shape, and that is the only question worth running this for.
"""

from __future__ import annotations

import json

from retail_agent.evals.compare import (
    Comparison,
    compare_runs,
    load_run,
    render_comparison,
)
from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import CaseResult, Gate


def result(case_id: str, outcome: Outcome, **kwargs) -> CaseResult:
    return CaseResult(case_id=case_id, outcome=outcome, **kwargs)


def gate(*results: CaseResult) -> Gate:
    passed = [r for r in results if r.outcome is Outcome.PASS]
    accuracy = len(passed) / len(results) if results else 0.0
    return Gate(passed=True, accuracy=accuracy, reason="", results=tuple(results))


def test_cases_both_arms_got_right_are_not_disagreements():
    left = gate(result("a", Outcome.PASS), result("b", Outcome.PASS))
    right = gate(result("a", Outcome.PASS), result("b", Outcome.PASS))

    assert compare_runs(left, right).disagreements == ()


def test_a_case_one_arm_got_wrong_is_a_disagreement():
    left = gate(result("a", Outcome.PASS), result("b", Outcome.PASS))
    right = gate(result("a", Outcome.PASS), result("b", Outcome.FAIL))

    disagreements = compare_runs(left, right).disagreements

    assert [d.case_id for d in disagreements] == ["b"]
    assert disagreements[0].left.outcome is Outcome.PASS
    assert disagreements[0].right.outcome is Outcome.FAIL


def test_two_arms_wrong_in_different_ways_still_disagree():
    """Both wrong is not agreement if they failed differently — one erroring and
    one answering the wrong number are different findings."""
    left = gate(result("a", Outcome.FAIL))
    right = gate(result("a", Outcome.ERROR))

    assert [d.case_id for d in compare_runs(left, right).disagreements] == ["a"]


def test_a_case_missing_from_one_run_is_reported_not_dropped():
    """A run that crashed partway would otherwise compare as a clean sweep of
    whatever it did finish."""
    left = gate(result("a", Outcome.PASS), result("b", Outcome.PASS))
    right = gate(result("a", Outcome.PASS))

    comparison = compare_runs(left, right)

    assert comparison.only_in_left == ("b",)
    assert comparison.only_in_right == ()


def test_cost_is_averaged_over_the_cases_that_ran():
    left = gate(
        result("a", Outcome.PASS, tokens_in=100, tokens_out=10, calls=2, seconds=1.0),
        result("b", Outcome.PASS, tokens_in=300, tokens_out=30, calls=4, seconds=3.0),
    )
    right = gate(result("a", Outcome.PASS, tokens_in=1000, tokens_out=100, calls=9))

    comparison = compare_runs(left, right)

    assert comparison.left.mean_tokens_in == 200
    assert comparison.left.mean_calls == 3
    assert comparison.right.mean_tokens_in == 1000


def test_an_empty_run_averages_to_zero_rather_than_dividing_by_zero():
    assert compare_runs(gate(), gate()).left.mean_tokens_in == 0


def test_a_report_written_by_one_run_can_be_read_back_for_comparison():
    """`compare` consumes the same `--json` files `eval` already writes. A
    second serialisation format would be a second thing to keep in step."""
    from retail_agent.evals.report import to_json

    original = gate(result("a", Outcome.PASS, tokens_in=100, calls=2))

    restored = load_run(json.loads(to_json(original)))

    assert restored.accuracy == original.accuracy
    assert restored.results[0].case_id == "a"
    assert restored.results[0].outcome is Outcome.PASS
    assert restored.results[0].tokens_in == 100


def test_the_rendered_report_names_both_arms_and_the_disagreeing_cases():
    left = gate(result("a", Outcome.PASS), result("b", Outcome.PASS))
    right = gate(result("a", Outcome.PASS), result("b", Outcome.FAIL))

    rendered = render_comparison(
        compare_runs(left, right), left_name="graph", right_name="react"
    )

    assert "graph" in rendered
    assert "react" in rendered
    assert "b" in rendered


def test_a_comparison_with_no_disagreements_says_so_explicitly():
    """Silence reads as a missing section. It has to say the arms agreed."""
    identical = gate(result("a", Outcome.PASS))

    rendered = render_comparison(compare_runs(identical, identical))

    assert "agree" in rendered.lower()


def test_comparison_is_a_value_not_a_printout():
    """So a caller can assert on it. The renderer is separate for the same
    reason `Gate` and `render_report` are."""
    assert isinstance(compare_runs(gate(), gate()), Comparison)
