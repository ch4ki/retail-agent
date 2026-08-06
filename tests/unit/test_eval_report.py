"""Reporting a run.

The report is what someone reads when the gate blocks a release, so it has to
say which cases failed and what they returned — not just a percentage.
"""

from __future__ import annotations

import json

from retail_agent.evals.gate import evaluate_gate
from retail_agent.evals.report import render_report, to_json
from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import CaseResult


def results():
    return [
        CaseResult(case_id="loyal-count", outcome=Outcome.FAIL, actual=1254, expected=5746,
                   detail="expected 5746, got 1254", sql="SELECT 1",
                   used_trios=("loyal-customers",)),
        CaseResult(case_id="total-users", outcome=Outcome.PASS, actual=100000, expected=100000),
        CaseResult(case_id="churn-count", outcome=Outcome.ERROR, detail="budget exhausted"),
    ]


def test_the_report_names_failing_cases_and_their_numbers():
    text = render_report(evaluate_gate(results(), threshold=0.9))

    assert "loyal-count" in text
    assert "1254" in text and "5746" in text


def test_the_report_separates_wrong_answers_from_no_answer():
    """Two different problems. An agent that degrades honestly needs a
    different fix from one that answers confidently and wrongly."""
    text = render_report(evaluate_gate(results(), threshold=0.9))

    assert "1 wrong" in text or "wrong: 1" in text.lower()
    assert "1 error" in text or "error: 1" in text.lower()


def test_a_passing_case_does_not_clutter_the_failure_list():
    text = render_report(evaluate_gate(results(), threshold=0.9))
    failures = text.split("Failures")[-1] if "Failures" in text else text

    assert "total-users" not in failures


def test_the_verdict_is_stated():
    blocked = render_report(evaluate_gate(results(), threshold=0.9))
    clean = render_report(evaluate_gate([results()[1]], threshold=0.9))

    assert "FAIL" in blocked.upper()
    assert "PASS" in clean.upper()


def test_json_output_round_trips_for_the_baseline():
    """The next run compares against this, so it has to be machine-readable and
    carry the accuracy it recorded."""
    gate = evaluate_gate(results(), threshold=0.9)

    parsed = json.loads(to_json(gate))

    assert parsed["accuracy"] == gate.accuracy
    assert parsed["passed"] is False
    assert len(parsed["results"]) == 3
    assert parsed["results"][0]["case_id"] == "loyal-count"


def test_json_records_the_outcome_as_a_readable_name():
    """`Outcome.FAIL` serialised as an enum repr is unreadable in a diff and
    unparseable by whatever reads the baseline back."""
    parsed = json.loads(to_json(evaluate_gate(results(), threshold=0.9)))

    assert parsed["results"][0]["outcome"] == "fail"


def test_a_pii_leak_is_impossible_to_miss_in_the_report():
    leaked = [CaseResult(case_id="oops", outcome=Outcome.PASS, pii_leaked=True)]

    text = render_report(evaluate_gate(leaked))

    assert "PII" in text
    assert "oops" in text


def test_numbers_are_rendered_plainly_not_as_library_reprs():
    """BigQuery returns numpy scalars, and "expected np.int64(5746)" in a
    failure report reads like a bug in the eval rather than in the agent."""
    import numpy as np

    results = [
        CaseResult(case_id="c", outcome=Outcome.FAIL, actual=np.int64(1), expected=np.int64(5746))
    ]

    text = render_report(evaluate_gate(results, threshold=0.9))

    assert "np.int64" not in text
    assert "5746" in text and "expected 5746" in text
