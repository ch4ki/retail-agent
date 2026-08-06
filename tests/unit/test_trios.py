"""The undefined-term rule.

This is the mechanism the design argues is the whole point of the Golden
Bucket: without it the agent picks a definition, writes clean SQL, and returns a
confident number nobody can trace back to a decision.
"""

import pytest

from retail_agent.knowledge.trios import (
    Trio,
    assumption_note,
    definitions_block,
    style_examples,
    undefined_terms,
    unresolved,
)


def trio(**overrides):
    base = dict(
        id="t1",
        question="Which customers churned last quarter?",
        sql="SELECT ...",
        report="Churn rose to 4.1%...",
        metric_definitions={
            "churn": "ordered in the prior 180 days, nothing in the trailing 90"
        },
        tags=("churn", "retention"),
    )
    base.update(overrides)
    return Trio(**base)


# --- detecting what the schema cannot settle ---


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("why did our churn rate spike last month?", ["churn"]),
        ("why are users in state X underspending?", ["underspending"]),
        ("who are our top customers?", ["top"]),
        ("which brands are performing well?", ["performing well"]),
        ("which customers are at risk?", ["at risk"]),
    ],
)
def test_the_briefs_own_questions_raise_a_term(question, expected):
    assert undefined_terms(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "what was total revenue in March 2024?",
        "how many orders shipped last week?",
        "compare revenue for brand X and brand Y",
        "what is the average order value by state?",
    ],
)
def test_a_question_the_schema_can_settle_raises_nothing(question):
    """A caveat on a question that did not need one is noise, and noise is how
    a warning stops being read."""
    assert undefined_terms(question) == []


def test_a_longer_phrase_wins_over_the_word_inside_it():
    assert undefined_terms("which brands are performing well?") == ["performing well"]


def test_terms_are_reported_in_the_order_they_were_asked():
    found = undefined_terms("show me top customers who are at risk")

    assert found == ["top", "at risk"]


def test_detection_is_case_insensitive():
    assert undefined_terms("Why did CHURN spike?") == ["churn"]


def test_a_term_inside_another_word_is_not_a_match():
    """"topic" is not "top"; "active" is, but "radioactive" is not."""
    assert undefined_terms("what topics do customers ask about?") == []
    assert undefined_terms("radioactive materials") == []


# --- resolving against retrieved trios ---


def test_a_term_a_trio_defines_is_resolved():
    assert unresolved("why did churn spike?", [trio()]) == []


def test_a_term_no_trio_defines_stays_unresolved():
    assert unresolved("who are our top customers?", [trio()]) == ["top"]


def test_no_trios_at_all_leaves_every_term_unresolved():
    assert unresolved("why did churn spike?", []) == ["churn"]


def test_resolution_is_case_insensitive():
    defined = trio(metric_definitions={"CHURN": "..."})

    assert unresolved("why did churn spike?", [defined]) == []


# --- what reaches the model ---


def test_definitions_are_injected_but_past_sql_is_not():
    """A previous query carries its own date filters and joins. Pasting it into
    a new question answers last quarter's question with this quarter's label."""
    block = definitions_block([trio()])

    assert "ordered in the prior 180 days" in block
    assert "SELECT" not in block


def test_definitions_are_deduplicated_across_trios():
    block = definitions_block([trio(), trio(id="t2")])

    assert block.count("churn:") == 1


def test_no_trios_means_no_definitions_block():
    assert definitions_block([]) == ""


def test_the_report_is_offered_as_a_shape_not_as_content():
    examples = style_examples([trio()])

    assert "Churn rose to 4.1%" in examples
    assert "not this content" in examples


def test_style_examples_are_capped():
    many = [trio(id=f"t{i}", report=f"Report {i}") for i in range(5)]

    assert style_examples(many, limit=2).count("---") == 1


# --- what the user is told ---


def test_the_assumption_note_names_the_term_and_the_judgement():
    note = assumption_note(["churn"])

    assert "churn" in note
    assert "window" in note, "says what specifically was undecided"


def test_the_note_asks_for_a_statement_not_a_refusal():
    """The agent should answer and say what it assumed. Refusing to answer a
    question an executive asked is not safety, it is unhelpfulness."""
    note = assumption_note(["top"])

    assert "do not refuse" in note.lower()


def test_no_terms_means_no_note():
    assert assumption_note([]) == ""
