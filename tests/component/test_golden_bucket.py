"""The Golden Bucket in the turn.

The failure this prevents is silent: the agent answers "why did churn spike?"
by inventing a definition, and nothing in the output shows a judgement was made.
"""

import pandas as pd

from retail_agent.agent.graph import build_graph, run_turn
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.knowledge.trios import Trio
from tests.component.conftest import FakeSource

ROWS = pd.DataFrame({"churned": [412]})


def turn(deps, question):
    return run_turn(build_graph(deps), user_id="dana", session_id="s1", question=question)


def analysis_replies(answer="Churn was 4.1%."):
    return [
        {"intent": "analyze"},
        {"steps": ["count churned customers"]},
        "SELECT COUNT(*) AS churned FROM orders",
        answer,
    ]


def test_a_matching_definition_reaches_the_sql_prompt(make_deps):
    """The agreed definition, not the stored query — that is the whole design
    argument, so it is asserted both ways."""
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "why did our churn rate spike last month?")

    sql_prompt = deps.llm.prompts[2]
    assert "trailing 90 days" in sql_prompt, "the definition was injected"
    assert "WITH active_before" not in sql_prompt, "the stored SQL was not"


def test_the_definition_also_reaches_synthesis(make_deps):
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "why did our churn rate spike last month?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "trailing 90 days" in synthesis


def test_an_undefined_term_makes_the_agent_state_its_assumption(make_deps):
    """No trio defines "at risk". The agent must answer and say what it assumed
    — refusing an executive's question is not safety, it is unhelpfulness.

    Uses a term the corpus deliberately leaves open. "loyal" was here until a
    trio defined it, which is the feature working: a term stops being assumed
    the moment an analyst settles it."""
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "which customers are at risk?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "at risk" in synthesis
    assert "no agreed definition" in synthesis
    assert "do not refuse" in synthesis.lower()


def test_a_defined_term_produces_no_assumption_note(make_deps):
    """A caveat on a question that did not need one is noise, and noise is how
    a warning stops being read."""
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "why did our churn rate spike last month?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "no agreed definition" not in synthesis


def test_a_question_with_no_business_terms_is_untouched(make_deps):
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "what was total revenue in March 2024?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "no agreed definition" not in synthesis


def test_no_corpus_still_flags_the_undefined_term(make_deps):
    """An empty bucket is the worst case for silent guessing, so the rule has
    to hold without any corpus at all."""
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))

    turn(deps, "why did our churn rate spike?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "no agreed definition" in synthesis
    assert "churn" in synthesis


def test_a_superseded_definition_is_never_used(make_deps):
    """Definitions change. A new question must use the current one."""
    old = Trio(
        id="old-churn", question="Which customers churned?", sql="SELECT 1",
        report="...", metric_definitions={"churn": "sixty days of silence"},
        tags=("churn",), superseded_by="churn-90",
    )
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": [old]})

    turn(deps, "why did our churn rate spike?")

    sql_prompt = deps.llm.prompts[2]
    assert "sixty days" not in sql_prompt


def test_the_sql_writer_is_told_to_choose_a_value_for_an_undefined_term(make_deps):
    """Live failure: asked "how many loyal customers", the model had no agreed
    threshold, wrote `HAVING COUNT(order_id) > @threshold`, and BigQuery
    rejected it three times. The synthesis prompt knew the term was undefined;
    the prompt that actually writes the SQL did not."""
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "which customers are at risk?")

    sql_prompt = deps.llm.prompts[2]
    assert "at risk" in sql_prompt
    assert "literal" in sql_prompt.lower(), "told to inline a concrete value"


def test_no_assumption_means_no_extra_instruction_in_the_sql_prompt(make_deps):
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "why did our churn rate spike last month?")

    sql_prompt = deps.llm.prompts[2]
    assert "no agreed definition" not in sql_prompt.lower()


def test_the_analysts_report_style_reaches_synthesis(make_deps):
    """§5.1 calls the `report` field "hard to specify and easy to demonstrate":
    split by cohort, compare to a baseline, close with numbered actions. It was
    stored, tested, and never sent to the model."""
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))
    deps = deps.__class__(**{**deps.__dict__, "trios": list(SEED_TRIOS)})

    turn(deps, "why did our churn rate spike last month?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "acquired through Email" in synthesis, "the example report was included"
    assert "not this content" in synthesis, "and framed as a shape to match"


def test_no_trio_means_no_style_block(make_deps):
    deps = make_deps(analysis_replies(), src=FakeSource(frames={"ok": ROWS}))

    turn(deps, "what was total revenue in March 2024?")

    synthesis = [p for p in deps.llm.prompts if "Query results" in p][-1]
    assert "not this content" not in synthesis
