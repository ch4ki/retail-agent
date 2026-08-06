"""Scoring an eval case.

Execution accuracy: does the agent's number match the reference number. Not SQL
string match — many different queries are correct, and the one the agent writes
is rarely the one an analyst would.
"""

from __future__ import annotations

import pytest

from retail_agent.evals.scoring import Outcome, compare


def test_an_exact_match_passes():
    assert compare(actual=5746, expected=5746).outcome is Outcome.PASS


def test_a_wrong_number_fails():
    """The failure this whole suite exists for: four live runs invented four
    different rules for "loyal" and returned 0, 1254 and 66133 against a
    verified 5746. Every path-based test passed through all of them."""
    result = compare(actual=1254, expected=5746)

    assert result.outcome is Outcome.FAIL
    assert "1254" in result.detail and "5746" in result.detail


def test_floating_point_noise_is_not_a_failure():
    """Revenue sums differ in the last bits depending on aggregation order."""
    assert compare(actual=1234567.8900001, expected=1234567.89).outcome is Outcome.PASS


def test_tolerance_is_relative_so_it_scales_with_the_number():
    """An absolute epsilon is either useless on millions or absurd on counts."""
    assert compare(actual=1_000_500, expected=1_000_000, tolerance=0.01).outcome is Outcome.PASS
    assert compare(actual=1_020_000, expected=1_000_000, tolerance=0.01).outcome is Outcome.FAIL


def test_zero_expected_is_not_a_division_by_zero():
    """"No orders matched" is a legitimate answer, so it must be scorable."""
    assert compare(actual=0, expected=0).outcome is Outcome.PASS
    assert compare(actual=3, expected=0).outcome is Outcome.FAIL


def test_a_missing_answer_is_an_error_not_a_wrong_answer():
    """The agent degrading (budget exhausted, guard rejection) is a different
    failure from confidently answering wrongly, and mixing them into one number
    would hide which is happening."""
    result = compare(actual=None, expected=5746)

    assert result.outcome is Outcome.ERROR


def test_ranked_rows_compare_by_order():
    """"Top 10 customers" is wrong if it returns the right ten in the wrong
    order — the ranking is the answer."""
    assert compare(actual=[3, 1, 2], expected=[3, 1, 2]).outcome is Outcome.PASS
    assert compare(actual=[1, 2, 3], expected=[3, 1, 2]).outcome is Outcome.FAIL


def test_row_comparison_tolerates_noise_in_each_value():
    assert compare(actual=[10.000001, 20.0], expected=[10.0, 20.0]).outcome is Outcome.PASS


def test_a_shorter_answer_than_the_reference_fails():
    """Asking for ten and getting three is not a pass."""
    assert compare(actual=[1, 2, 3], expected=[1, 2, 3, 4, 5]).outcome is Outcome.FAIL


def test_a_number_that_cannot_be_read_is_an_error():
    """The agent answered in prose with no usable figure. Not a wrong number —
    there is no number."""
    assert compare(actual="quite a lot", expected=5746).outcome is Outcome.ERROR


@pytest.mark.parametrize("actual", [float("nan"), float("inf")])
def test_non_finite_values_never_pass(actual):
    """NaN == NaN is False, and `abs(nan - x) < tol` is False, so a naive
    implementation reports FAIL rather than ERROR and buries the real problem."""
    assert compare(actual=actual, expected=5746).outcome is Outcome.ERROR
