"""What a trio carries into the prompt, and what the answer has to disclose.

Detecting the undefined term used to live here too — a regex over nineteen
words, with tests pinning which ones it found. Both are gone: the model decides
what it cannot settle by calling `ask_for_definitions`, so there is no word list
to assert against. What survives is everything downstream of that decision.
"""

from retail_agent.knowledge.trios import (
    Trio,
    assumption_note,
    definitions_block,
    sql_assumption_note,
    style_examples,
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


def test_the_assumption_note_names_the_term_and_demands_the_rule():
    note = assumption_note(["churn"])

    assert "churn" in note
    assert "concrete rule" in note, "the rule applied, not a gloss on the term"


def test_the_note_asks_for_a_statement_not_a_refusal():
    """The agent should answer and say what it assumed. Refusing to answer a
    question an executive asked is not safety, it is unhelpfulness."""
    note = assumption_note(["top"])

    assert "do not refuse" in note.lower()


def test_no_terms_means_no_note():
    assert assumption_note([]) == ""


def test_a_term_from_outside_the_old_word_list_still_gets_a_note():
    """The terms come from the executive's own question now rather than from a
    fixed dict, so any word has to produce a note instead of a KeyError."""
    note = assumption_note(["LGB"])

    assert "LGB" in note
    assert "do not refuse" in note.lower()


def test_the_sql_note_names_a_term_from_outside_the_old_word_list():
    note = sql_assumption_note(["LGB"])

    assert "LGB" in note
    assert "literal" in note.lower(), "still warns off bind parameters"
