"""Running a case.

The runner takes two seams — ask the agent, execute reference SQL — so the
harness is testable with no BigQuery, no LLM and no key. A suite that can only
be verified by the person holding the credentials is a suite nobody else trusts.
"""

from __future__ import annotations

import pytest

from retail_agent.evals.runner import AgentAnswer, run_case
from retail_agent.evals.scoring import Outcome
from retail_agent.evals.types import EvalCase

CASE = EvalCase(
    id="loyal-count",
    question="How many loyal customers do we have?",
    reference_sql="SELECT 5746 AS n",
    required_definitions=("loyal",),
)


def answer(value, **kwargs) -> AgentAnswer:
    defaults = dict(text=f"There are {value}.", rows=[[value]], columns=("n",), sql="SELECT 1")
    return AgentAnswer(**{**defaults, **kwargs})


def test_a_correct_answer_passes():
    result = run_case(
        CASE, ask=lambda _q: answer(5746), execute=lambda _sql: [[5746]]
    )

    assert result.outcome is Outcome.PASS
    assert result.case_id == "loyal-count"


def test_a_wrong_answer_fails_and_records_both_numbers():
    """The whole point. A path-based test cannot tell these two runs apart."""
    result = run_case(
        CASE, ask=lambda _q: answer(1254), execute=lambda _sql: [[5746]]
    )

    assert result.outcome is Outcome.FAIL
    assert result.actual == 1254
    assert result.expected == 5746


def test_the_number_is_read_from_the_result_rows_not_the_prose():
    """Parsing the narrative would measure the model's phrasing. The rows are
    what the SQL actually returned."""
    result = run_case(
        CASE,
        ask=lambda _q: answer(5746, text="Roughly six thousand, give or take."),
        execute=lambda _sql: [[5746]],
    )

    assert result.outcome is Outcome.PASS


def test_pii_in_the_answer_is_recorded_even_when_the_number_is_right():
    """These are independent failures. A correct answer that leaks an email
    address is still a blocking problem, and scoring it as a pass would hide
    it."""
    result = run_case(
        CASE,
        ask=lambda _q: answer(5746, text="Top spender: alice@example.com"),
        execute=lambda _sql: [[5746]],
    )

    assert result.outcome is Outcome.PASS
    assert result.pii_leaked


def test_an_agent_crash_is_an_error_not_a_crashed_run():
    """One broken case must not take the other thirty-nine with it."""

    def explode(_question):
        raise RuntimeError("provider is down")

    result = run_case(CASE, ask=explode, execute=lambda _sql: [[5746]])

    assert result.outcome is Outcome.ERROR
    assert "provider is down" in result.detail


def test_a_broken_reference_query_is_an_error_against_the_suite():
    """Distinguishable from the agent being wrong: this is the eval's own fault
    and must never be reported as an agent failure."""

    def explode(_sql):
        raise RuntimeError("bad reference SQL")

    result = run_case(CASE, ask=lambda _q: answer(5746), execute=explode)

    assert result.outcome is Outcome.ERROR
    assert "reference" in result.detail.lower()


def test_an_empty_reference_result_is_an_error():
    result = run_case(CASE, ask=lambda _q: answer(5746), execute=lambda _sql: [])

    assert result.outcome is Outcome.ERROR


def test_a_ranked_case_compares_the_whole_column_in_order():
    case = EvalCase(
        id="top-3",
        question="Top 3 customers by spend",
        reference_sql="SELECT id FROM t",
        ranked=True,
    )

    result = run_case(
        case,
        ask=lambda _q: AgentAnswer(
            text="", rows=[[7], [4], [9]], columns=("id",), sql=""
        ),
        execute=lambda _sql: [[7], [4], [9]],
    )

    assert result.outcome is Outcome.PASS


def test_a_ranked_case_fails_when_the_order_differs():
    case = EvalCase(id="top-3", question="q", reference_sql="s", ranked=True)

    result = run_case(
        case,
        ask=lambda _q: AgentAnswer(text="", rows=[[4], [7], [9]], columns=("id",), sql=""),
        execute=lambda _sql: [[7], [4], [9]],
    )

    assert result.outcome is Outcome.FAIL


def test_an_agent_that_returned_no_rows_is_an_error_not_a_wrong_number():
    result = run_case(
        CASE,
        ask=lambda _q: AgentAnswer(text="I could not answer.", rows=[], columns=(), sql=""),
        execute=lambda _sql: [[5746]],
    )

    assert result.outcome is Outcome.ERROR


def test_the_result_carries_what_is_needed_to_debug_it():
    """A failing case that reports only "wrong" costs an hour of re-running by
    hand to find out which SQL and which definitions produced it."""
    result = run_case(
        CASE,
        ask=lambda _q: answer(1254, sql="SELECT COUNT(*) FROM x", trios=("loyal-customers",)),
        execute=lambda _sql: [[5746]],
    )

    assert result.sql == "SELECT COUNT(*) FROM x"
    assert result.used_trios == ("loyal-customers",)
    assert result.answer


def test_tolerance_from_the_case_is_applied():
    case = EvalCase(
        id="revenue", question="q", reference_sql="s", tolerance=0.01
    )

    result = run_case(
        case, ask=lambda _q: answer(1_000_500), execute=lambda _sql: [[1_000_000]]
    )

    assert result.outcome is Outcome.PASS


@pytest.mark.parametrize("rows", [[[None]], [["n/a"]]])
def test_an_unreadable_agent_value_is_an_error(rows):
    result = run_case(
        CASE,
        ask=lambda _q: AgentAnswer(text="", rows=rows, columns=("n",), sql=""),
        execute=lambda _sql: [[5746]],
    )

    assert result.outcome is Outcome.ERROR


# --- running the whole suite ---


def test_the_suite_runs_every_case_and_gates_the_result():
    from retail_agent.evals.runner import run_suite

    cases = [
        EvalCase(id="a", question="q", reference_sql="s"),
        EvalCase(id="b", question="q", reference_sql="s"),
    ]

    gate = run_suite(cases, ask=lambda _q: answer(1), execute=lambda _s: [[1]])

    assert gate.accuracy == 1.0
    assert [r.case_id for r in gate.results] == ["a", "b"]


def test_one_exploding_case_does_not_stop_the_others():
    """Forty cases and one provider hiccup must still produce a report."""
    from retail_agent.evals.runner import run_suite

    def flaky(question):
        if question == "boom":
            raise RuntimeError("provider down")
        return answer(1)

    cases = [
        EvalCase(id="ok-1", question="fine", reference_sql="s"),
        EvalCase(id="bad", question="boom", reference_sql="s"),
        EvalCase(id="ok-2", question="fine", reference_sql="s"),
    ]

    gate = run_suite(cases, ask=flaky, execute=lambda _s: [[1]])

    assert len(gate.results) == 3
    assert gate.accuracy == pytest.approx(2 / 3)


def test_progress_is_reported_as_it_goes():
    """A live run is forty LLM turns against BigQuery — minutes of silence
    otherwise, and no way to tell a slow case from a hung one."""
    from retail_agent.evals.runner import run_suite

    seen = []
    run_suite(
        [EvalCase(id="a", question="q", reference_sql="s")],
        ask=lambda _q: answer(1),
        execute=lambda _s: [[1]],
        on_case=lambda result: seen.append(result.case_id),
    )

    assert seen == ["a"]


def test_the_baseline_is_passed_through_to_the_gate():
    from retail_agent.evals.runner import run_suite

    cases = [EvalCase(id=str(i), question="q", reference_sql="s") for i in range(20)]

    gate = run_suite(
        cases,
        ask=lambda _q: answer(2),  # every case wrong
        execute=lambda _s: [[1]],
        threshold=0.0,
        baseline=0.95,
    )

    assert not gate.passed
    assert "regress" in gate.reason.lower()


def test_a_scalar_case_that_returns_many_rows_says_so():
    """Seen on the first live run: for "how many loyal customers" the agent
    wrote `GROUP BY u.id`, which returns one row per customer each containing
    1, and then narrated "20 customers". Reporting only "got 1" sends the
    reader looking for an off-by-one instead of a wrong shape."""
    result = run_case(
        CASE,
        ask=lambda _q: AgentAnswer(
            text="", rows=[[1]] * 500, columns=("n",), sql="SELECT ... GROUP BY u.id"
        ),
        execute=lambda _sql: [[5746]],
    )

    assert result.outcome is Outcome.FAIL
    assert "500 rows" in result.detail


def test_a_single_row_scalar_answer_reports_plainly():
    result = run_case(CASE, ask=lambda _q: answer(1254), execute=lambda _sql: [[5746]])

    assert "rows" not in result.detail


# --- picking the right column out of the agent's answer ---


def test_the_named_answer_column_is_used_when_the_agent_returns_extras():
    """Live: asked which month of 2023 was busiest, the agent returned
    (year, month, total_orders) with the right month in it. Taking column 0
    scored 2023 against an expected 12 and called a correct answer wrong."""
    case = EvalCase(
        id="busiest-month",
        question="q",
        reference_sql="s",
        answer_column="month",
    )

    result = run_case(
        case,
        ask=lambda _q: AgentAnswer(
            text="", rows=[[2023, 12, 1580]], columns=("year", "month", "total_orders")
        ),
        execute=lambda _sql: [[12]],
    )

    assert result.outcome is Outcome.PASS


def test_a_missing_named_column_falls_back_to_the_first():
    """The name is a hint, not a contract — the agent chooses its own aliases."""
    case = EvalCase(id="c", question="q", reference_sql="s", answer_column="month")

    result = run_case(
        case,
        ask=lambda _q: AgentAnswer(text="", rows=[[12]], columns=("m",)),
        execute=lambda _sql: [[12]],
    )

    assert result.outcome is Outcome.PASS


def test_the_column_name_match_ignores_case():
    case = EvalCase(id="c", question="q", reference_sql="s", answer_column="Month")

    result = run_case(
        case,
        ask=lambda _q: AgentAnswer(text="", rows=[[2023, 12]], columns=("year", "month")),
        execute=lambda _sql: [[12]],
    )

    assert result.outcome is Outcome.PASS


def test_without_a_named_column_the_first_is_still_used():
    """Unchanged for the 40-odd cases that return a single column."""
    result = run_case(CASE, ask=lambda _q: answer(5746), execute=lambda _sql: [[5746]])

    assert result.outcome is Outcome.PASS


def test_a_ranked_case_ignores_the_answer_column():
    """Ranked cases compare a whole ordered column; the hint does not apply."""
    case = EvalCase(id="c", question="q", reference_sql="s", ranked=True,
                    answer_column="brand")

    result = run_case(
        case,
        ask=lambda _q: AgentAnswer(text="", rows=[["a"], ["b"]], columns=("brand",)),
        execute=lambda _sql: [["a"], ["b"]],
    )

    assert result.outcome is Outcome.PASS
