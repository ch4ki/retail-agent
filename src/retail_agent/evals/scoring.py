"""Comparing an answer to the reference, and deciding what that means.

Execution accuracy, not SQL string match: many different queries are correct,
and the one the agent writes is rarely the one an analyst would write. What
matters is whether the number it arrived at is the number.

Three outcomes rather than two. "Wrong" and "could not answer" are different
failures — one is the agent being confidently incorrect, the other is it
degrading honestly — and averaging them into a single score would hide which of
the two is happening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Relative, because an absolute epsilon is either useless on revenue in the
# millions or absurdly loose on a customer count.
DEFAULT_TOLERANCE = 1e-6


class Outcome(Enum):
    PASS = "pass"
    FAIL = "fail"  # answered, and the number is wrong
    ERROR = "error"  # produced no comparable number at all


@dataclass(frozen=True)
class Result:
    outcome: Outcome
    detail: str = ""


def compare(*, actual: Any, expected: Any, tolerance: float = DEFAULT_TOLERANCE) -> Result:
    """Score one answer against the reference."""
    if actual is None:
        return Result(Outcome.ERROR, "the agent produced no answer")

    if isinstance(expected, (list, tuple)):
        return _compare_rows(actual, expected, tolerance)

    return _compare_scalar(actual, expected, tolerance)


def _compare_scalar(actual: Any, expected: Any, tolerance: float) -> Result:
    expected_number = _as_number(expected)

    # Plenty of real questions answer with a name rather than a number — which
    # brand earns most, which category, which country. Demanding a number on
    # both sides made those unscoreable, and two cases the agent answered
    # *correctly* were reported as ERROR because "Calvin Klein" is not a float.
    if expected_number is None:
        if str(actual).strip() == str(expected).strip():
            return Result(Outcome.PASS)
        return Result(Outcome.FAIL, f"expected {expected!r}, got {actual!r}")

    actual_number = _as_number(actual)
    if actual_number is None:
        # The reference is a number and the agent produced none. That is "no
        # answer", not "the wrong name".
        return Result(Outcome.ERROR, f"no number could be read from {actual!r}")

    if _close(actual_number, expected_number, tolerance):
        return Result(Outcome.PASS)
    return Result(Outcome.FAIL, f"expected {expected_number:g}, got {actual_number:g}")


def _compare_rows(actual: Any, expected: Any, tolerance: float) -> Result:
    if not isinstance(actual, (list, tuple)):
        return Result(Outcome.ERROR, f"expected {len(expected)} rows, got {actual!r}")

    if len(actual) != len(expected):
        return Result(
            Outcome.FAIL, f"expected {len(expected)} rows, got {len(actual)}"
        )

    # Order-sensitive on purpose. For "top 10 customers" the ranking *is* the
    # answer: the right ten in the wrong order is a wrong answer.
    for position, (got, want) in enumerate(zip(actual, expected)):
        got_number, want_number = _as_number(got), _as_number(want)
        if got_number is None or want_number is None:
            if got != want:
                return Result(Outcome.FAIL, f"row {position}: expected {want!r}, got {got!r}")
            continue
        if not _close(got_number, want_number, tolerance):
            return Result(
                Outcome.FAIL, f"row {position}: expected {want_number:g}, got {got_number:g}"
            )

    return Result(Outcome.PASS)


def _close(actual: float, expected: float, tolerance: float) -> bool:
    # Relative to the expected value, with an absolute fallback so that an
    # expected zero — "no orders matched", a legitimate answer — is comparable
    # rather than a division by zero.
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def _as_number(value: Any) -> float | None:
    """A float, or None when there is no number here.

    Bools are excluded deliberately: `True` is `1` in Python, and scoring a
    boolean as the number one would silently pass a nonsense answer.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip().replace(",", ""))
        except (TypeError, ValueError):
            return None

    # NaN and infinity compare False against everything, so a naive
    # implementation would report FAIL — "the number is wrong" — when the truth
    # is that there is no usable number.
    return number if math.isfinite(number) else None
