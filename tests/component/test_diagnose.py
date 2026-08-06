"""An empty result is not an error, and that is exactly why it needs its own
budget. Reproduced live: asked to compare `Levis`, the agent ran valid SQL that
matched nothing (the brand is stored as `Levi's`), got zero rows, and reported
that as the answer.
"""

import pandas as pd

from retail_agent.agent.graph import build_graph, run_turn
from tests.component.conftest import FakeSource

ROWS = pd.DataFrame({"id": [1, 2], "spend": [100, 90]})


def turn(graph, question="revenue for brand Levis"):
    return run_turn(graph, user_id="dana", session_id="s1", question=question)


def test_an_empty_result_is_retried_once_and_can_succeed(make_deps):
    source = FakeSource(frames={"ok": ROWS}, empty_for={"Levis"})
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue for brand Levis"]},
            "SELECT SUM(spend) AS spend FROM users WHERE brand = 'Levis'",  # empty
            "SELECT SUM(spend) AS spend FROM users WHERE brand LIKE '%Levi%'",
            "Revenue was $190.",
        ],
        src=source,
    )

    state = turn(build_graph(deps))

    assert len(source.executed) == 2, "the empty result should have been diagnosed"
    assert state["status"] == "ok"
    assert state["frames"]["step_1"].row_count == 2


def test_diagnosis_does_not_spend_the_repair_budget(make_deps):
    """Two separate failure modes. Spending syntax retries on an empty result
    starves the budget that exists for genuinely broken SQL."""
    source = FakeSource(frames={"ok": ROWS}, empty_for={"Levis"})
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue for brand Levis"]},
            "SELECT SUM(spend) FROM users WHERE brand = 'Levis'",
            "SELECT SUM(spend) FROM users WHERE brand LIKE '%Levi%'",
            "Revenue was $190.",
        ],
        src=source,
    )

    state = turn(build_graph(deps))

    assert state["repair_budget"] == deps.settings.repair_budget, "untouched"
    assert state["diagnose_budget"] == 0, "the diagnosis budget paid for it"


def test_a_still_empty_result_is_accepted_rather_than_looping(make_deps):
    """Sometimes "no orders matched" is the true answer. One diagnosis, then
    the emptiness is reported as a finding."""
    source = FakeSource(frames={"ok": ROWS}, empty_for={"Levis"})
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue for brand Levis"]},
            "SELECT 1 FROM users WHERE brand = 'Levis'",
            "SELECT 2 FROM users WHERE brand = 'Levis'",  # still nothing
            "No revenue was recorded for that brand.",
        ],
        src=source,
    )

    state = turn(build_graph(deps))

    assert len(source.executed) == 2, "diagnosed once, then accepted"
    assert state["frames"]["step_1"].row_count == 0


def test_the_diagnosis_tells_the_model_what_to_check(make_deps):
    source = FakeSource(frames={"ok": ROWS}, empty_for={"Levis"})
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue for brand Levis"]},
            "SELECT SUM(spend) FROM users WHERE brand = 'Levis'",
            "SELECT SUM(spend) FROM users WHERE brand LIKE '%Levi%'",
            "Revenue was $190.",
        ],
        src=source,
    )

    turn(build_graph(deps))

    from retail_agent.agent.nodes.diagnose import EMPTY_RESULT_DIAGNOSIS

    redraft = deps.llm.prompts[3]
    # Assert against the constant, not a phrase: rewording the guidance should
    # not fail this, but failing to hand it to the model must.
    assert EMPTY_RESULT_DIAGNOSIS in redraft
    # The specific advice that made the live retry succeed. A whole-word LIKE
    # cannot match "Levi's", so the model has to be told to shorten the fragment.
    assert "FRAGMENT" in EMPTY_RESULT_DIAGNOSIS


def test_a_non_empty_result_is_never_diagnosed(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["top customers"]},
            "SELECT id, spend FROM users",
            "Your top customer spent $100.",
        ],
        src=source,
    )

    state = turn(build_graph(deps), "top customers")

    assert len(source.executed) == 1
    assert state["diagnose_budget"] == deps.settings.diagnose_budget


def test_an_aggregate_over_no_rows_is_diagnosed_too(make_deps):
    """The shape the live failure actually took. `SUM(sale_price)` where the
    brand matches nothing returns ONE row holding NULL, not zero rows — so a
    trigger that only checks row_count never fires on the case it exists for."""
    source = FakeSource(frames={"ok": ROWS}, null_aggregate_for={"'Levis'"})
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue for brand Levis in 2024"]},
            "SELECT SUM(sale_price) AS total FROM order_items WHERE brand = 'Levis'",
            "SELECT SUM(sale_price) AS total FROM order_items WHERE brand LIKE '%Levi%'",
            "Revenue was $190.",
        ],
        src=source,
    )

    state = turn(build_graph(deps))

    assert len(source.executed) == 2, "one NULL row is still nothing found"
    assert state["diagnose_budget"] == 0
    assert state["frames"]["step_1"].row_count == 2


def test_a_row_with_some_values_is_not_treated_as_empty(make_deps, source):
    """A real result that happens to contain a NULL column must not be
    diagnosed — that would burn the budget on a perfectly good answer."""
    import pandas as pd

    partial = FakeSource(
        frames={"ok": pd.DataFrame({"brand": ["Levi's"], "revenue": [None]})}
    )
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue by brand"]},
            "SELECT brand, SUM(spend) AS revenue FROM users GROUP BY brand",
            "Levi's has no recorded revenue.",
        ],
        src=partial,
    )

    state = turn(build_graph(deps))

    assert len(partial.executed) == 1
    assert state["diagnose_budget"] == deps.settings.diagnose_budget
