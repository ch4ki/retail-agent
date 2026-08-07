"""Multi-step plans must not fabricate their own inputs.

The first live eval run found the agent doing exactly that. Asked for an average
age it executed:

    SELECT AVG(age) FROM (SELECT 54 AS age UNION ALL SELECT 25 ... )
    /* Add the remaining 95 rows here */

The cause was not a bad prompt rule. Step 2's prompt rendered step 1's frame as
markdown truncated to five rows under the heading "Results already gathered in
this analysis". The model had no way to *reference* that data from SQL, so it
copied the rows it could see and asked for the rest in a comment. It passed the
guard, executed cleanly, triggered no repair, and the narrative was confident.

The fix is structural rather than another rule: a later step is never shown rows
it might copy. It is shown the earlier step's *query*. A model cannot inline
data it was never given.
"""

from __future__ import annotations

from retail_agent.agent.nodes.sql import draft_sql_node
from retail_agent.agent.state import (
    AnalysisStep,
    MaskedFrame,
    fresh_scratch,
    new_turn_state,
)


def two_step_state(*, frame: MaskedFrame, first_sql: str):
    """A turn where step 1 has run and step 2 is about to be drafted."""
    state = new_turn_state(user_id="dana", session_id="s1", question="average age")
    state.update(fresh_scratch(repair_budget=2))
    state["plan"] = [
        AnalysisStep(id="step_1", question="fetch the ages", sql=first_sql),
        AnalysisStep(id="step_2", question="average them"),
    ]
    state["step_index"] = 1
    state["frames"] = {"step_1": frame}
    return state


def many_rows(n=100) -> MaskedFrame:
    """A frame whose stored rows are a truncated sample of a larger result —
    the shape that produced the fabrication."""
    return MaskedFrame(
        columns=("age",),
        rows=tuple((age,) for age in (54, 25, 36, 41, 35)),
        row_count=n,
        redactions=0,
    )


def prompt_sent_to(deps) -> str:
    return "\n".join(deps.llm.prompts)


def test_a_later_step_is_never_shown_rows_it_could_copy(make_deps):
    """The structural guarantee. Everything else here follows from it."""
    deps = make_deps(["SELECT AVG(age) AS avg_age FROM users"])
    state = two_step_state(
        frame=many_rows(), first_sql="SELECT age FROM users LIMIT 500"
    )

    draft_sql_node(state, deps)

    prompt = prompt_sent_to(deps)
    assert "54" not in prompt, "step 1's row values reached the prompt"
    assert "25" not in prompt


def test_a_later_step_is_shown_the_earlier_query_instead(make_deps):
    """So it can compose in SQL, which is the only way to reach all the rows."""
    deps = make_deps(["SELECT AVG(age) AS avg_age FROM users"])
    state = two_step_state(
        frame=many_rows(), first_sql="SELECT age FROM users LIMIT 500"
    )

    draft_sql_node(state, deps)

    assert "SELECT age FROM users" in prompt_sent_to(deps)


def test_the_prompt_says_how_many_rows_there_really_were(make_deps):
    """Five stored rows standing for a hundred is what made truncation
    invisible. The count is what tells the model a sample would be wrong."""
    deps = make_deps(["SELECT 1 AS n"])
    state = two_step_state(frame=many_rows(100), first_sql="SELECT age FROM users")

    draft_sql_node(state, deps)

    assert "100" in prompt_sent_to(deps)


def test_a_single_scalar_result_is_still_given_as_a_value(make_deps):
    """A 1x1 result cannot be truncated, so inlining it is correct and useful —
    "the average is 41, now find who is above it" needs the number itself."""
    deps = make_deps(["SELECT id FROM users WHERE age > 41"])
    scalar = MaskedFrame(columns=("avg_age",), rows=((41,),), row_count=1, redactions=0)
    state = two_step_state(frame=scalar, first_sql="SELECT AVG(age) FROM users")

    draft_sql_node(state, deps)

    assert "41" in prompt_sent_to(deps)


def test_a_single_row_of_several_columns_is_given_as_values(make_deps):
    """Still unambiguous and still untruncatable."""
    deps = make_deps(["SELECT 1 AS n"])
    row = MaskedFrame(
        columns=("brand", "revenue"),
        rows=(("Calvin Klein", 155865),),
        row_count=1,
        redactions=0,
    )
    state = two_step_state(frame=row, first_sql="SELECT brand, revenue FROM x")

    draft_sql_node(state, deps)

    assert "Calvin Klein" in prompt_sent_to(deps)


def test_a_first_step_gets_no_prior_results_section(make_deps):
    deps = make_deps(["SELECT id FROM users LIMIT 5"])
    state = new_turn_state(user_id="dana", session_id="s1", question="q")
    state.update(fresh_scratch(repair_budget=2))
    state["plan"] = [AnalysisStep(id="step_1", question="q")]

    draft_sql_node(state, deps)

    assert "already gathered" not in prompt_sent_to(deps).lower()


def test_a_step_whose_query_is_unknown_is_not_described_as_referenceable(make_deps):
    """Defensive: a frame with no recorded SQL cannot be composed against, and
    telling the model to reference it would invite an invented table name."""
    deps = make_deps(["SELECT 1 AS n"])
    state = two_step_state(frame=many_rows(), first_sql=None)

    draft_sql_node(state, deps)

    prompt = prompt_sent_to(deps)
    assert "54" not in prompt, "still must not leak rows"


def test_the_earlier_query_is_handed_over_without_its_display_limit(make_deps):
    """The guard appends a LIMIT so a result set stays printable. That bound is
    about display, not meaning — and handing it to the next step made the model
    faithfully reproduce it:

        SELECT AVG(age) FROM (SELECT age FROM users LIMIT 100)

    which averages a hundred rows and calls it the average age. The truncation
    that used to happen in the prompt simply moved into the SQL.
    """
    deps = make_deps(["SELECT AVG(age) AS avg_age FROM users"])
    state = two_step_state(
        frame=many_rows(), first_sql="SELECT age FROM users LIMIT 100"
    )

    draft_sql_node(state, deps)

    prompt = prompt_sent_to(deps)
    assert "SELECT age FROM users" in prompt
    assert "LIMIT 100" not in prompt


def test_a_limit_the_question_asked_for_is_kept(make_deps):
    """Only the outermost bound is removed. "Top 10 customers" means ten, and
    stripping that would change what the earlier step actually computed."""
    deps = make_deps(["SELECT 1 AS n"])
    state = two_step_state(
        frame=many_rows(),
        first_sql="SELECT user_id FROM (SELECT user_id FROM orders LIMIT 10) LIMIT 500",
    )

    draft_sql_node(state, deps)

    assert "LIMIT 10" in prompt_sent_to(deps)


def test_a_sampled_earlier_result_reports_its_true_size(make_deps):
    """The count is exact even when only some rows were fetched, so a later
    step sizes its work against the real number rather than against a cap."""
    deps = make_deps(["SELECT 1 AS n"])
    frame = MaskedFrame(
        columns=("user_id",),
        rows=((1,), (2,)),
        row_count=5823,
        redactions=0,
        truncated=True,
    )
    state = two_step_state(frame=frame, first_sql="SELECT user_id FROM orders")

    draft_sql_node(state, deps)

    prompt = prompt_sent_to(deps)
    assert "5823 rows" in prompt
    assert "only the first 2 shown" in prompt
