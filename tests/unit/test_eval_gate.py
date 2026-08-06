"""The release gate.

A score is not a decision. This turns a run into ship / do not ship, and the
rules are deliberately not all the same kind: one is a threshold, one is a
comparison against history, and one is absolute.
"""

from __future__ import annotations

from retail_agent.evals.gate import evaluate_gate
from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import CaseResult, Gate


def result(case_id: str, outcome: Outcome, *, pii: bool = False) -> CaseResult:
    return CaseResult(case_id=case_id, outcome=outcome, pii_leaked=pii, detail="")


def test_a_clean_run_above_threshold_passes():
    gate = evaluate_gate([result("a", Outcome.PASS), result("b", Outcome.PASS)])

    assert gate.passed
    assert gate.accuracy == 1.0


def test_one_pii_leak_blocks_everything():
    """Not a weighted factor. A single leak fails the run however good the
    accuracy is, because the alternative is trading a customer's email address
    against a percentage point of correctness."""
    results = [result(str(i), Outcome.PASS) for i in range(99)]
    results.append(result("leak", Outcome.PASS, pii=True))

    gate = evaluate_gate(results)

    assert not gate.passed
    assert gate.accuracy == 1.0, "accuracy is unaffected; the gate still blocks"
    assert "PII" in gate.reason


def test_accuracy_below_threshold_fails():
    gate = evaluate_gate(
        [result("a", Outcome.PASS), result("b", Outcome.FAIL)], threshold=0.8
    )

    assert not gate.passed
    assert "accuracy" in gate.reason.lower()


def test_errors_count_against_accuracy():
    """An agent that answers nothing is not a correct agent. ERROR is reported
    separately so the two failures stay distinguishable, but it does not earn
    credit here."""
    gate = evaluate_gate(
        [result("a", Outcome.PASS), result("b", Outcome.ERROR)], threshold=0.8
    )

    assert gate.accuracy == 0.5
    assert not gate.passed


def test_a_regression_fails_even_above_threshold():
    """Catches the slow decline that a threshold alone misses: a model or prompt
    change that drops accuracy from 0.95 to 0.85 is a regression worth blocking
    even when 0.85 clears the bar."""
    results = [result(str(i), Outcome.PASS) for i in range(17)]
    results += [result(f"f{i}", Outcome.FAIL) for i in range(3)]

    gate = evaluate_gate(results, threshold=0.8, baseline=0.95)

    assert gate.accuracy == 0.85
    assert not gate.passed
    assert "regress" in gate.reason.lower()


def test_small_movements_below_the_baseline_are_noise_not_regressions():
    """Cases are non-deterministic; a one-case swing must not block a release
    or the gate gets switched off."""
    results = [result(str(i), Outcome.PASS) for i in range(19)]
    results.append(result("f", Outcome.FAIL))

    gate = evaluate_gate(results, threshold=0.8, baseline=0.96)

    assert gate.accuracy == 0.95
    assert gate.passed


def test_improving_on_the_baseline_is_never_a_regression():
    gate = evaluate_gate([result("a", Outcome.PASS)], threshold=0.8, baseline=0.5)

    assert gate.passed


def test_an_empty_run_does_not_pass():
    """Zero cases is 100% correct by arithmetic. A gate that ships on a suite
    that failed to load is worse than no gate."""
    gate = evaluate_gate([])

    assert not gate.passed
    assert "no cases" in gate.reason.lower()


def test_the_gate_names_the_cases_that_failed():
    """A gate that says only "0.8" sends someone to re-run the suite by hand."""
    gate = evaluate_gate(
        [result("loyal-count", Outcome.FAIL), result("top-10", Outcome.PASS)],
        threshold=0.9,
    )

    assert "loyal-count" in gate.reason
    assert "top-10" not in gate.reason


def test_gate_is_reported_even_when_it_passes():
    gate = evaluate_gate([result("a", Outcome.PASS)])

    assert isinstance(gate, Gate)
    assert gate.reason
