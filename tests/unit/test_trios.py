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
    lookup_definition,
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


def test_a_term_the_executive_overrode_is_left_out_of_the_corpus_block():
    """Where the executive has their own definition, rendering the corpus one
    beside it hands the model two meanings and lets it pick. The overridden
    term is withheld; the personal block carries the one in force."""
    block = definitions_block([trio()], except_for={"churn"})

    assert block == ""


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


# --- looking a phrase up ---


LOYAL = {"loyal": "three or more completed orders"}


def test_the_executives_phrase_finds_the_bare_term():
    assert lookup_definition(LOYAL, "loyal customers") == LOYAL["loyal"]


def test_a_word_merely_contained_does_not_settle_the_phrase():
    assert lookup_definition(LOYAL, "disloyal customers") is None


def test_a_negated_phrase_is_not_settled_by_the_positive_term():
    """"not loyal" answered with loyal's definition computes the loyal cohort
    for a question about its complement — silently, since nothing pauses."""
    assert lookup_definition(LOYAL, "not loyal customers") is None
    assert lookup_definition(LOYAL, "customers who are not loyal") is None


def test_an_unknown_qualifier_fails_closed_rather_than_guessing():
    """"least" and "semi" change what the phrase means. A word this function
    does not recognise must leave the phrase open — the cost is a question
    that was not strictly needed, never a silently wrong cohort."""
    assert lookup_definition(LOYAL, "least loyal customers") is None
    assert lookup_definition(LOYAL, "semi loyal customers") is None


def test_punctuation_glued_to_a_word_does_not_hide_the_term():
    """The model passes the executive's words verbatim — including the question
    mark and the quotes they typed."""
    assert lookup_definition({"churn": "no order in 90 days"}, "churn?") is not None
    assert lookup_definition({"power users": "ten or more orders"}, '"power users"') is not None


def test_a_phrase_holding_two_defined_terms_keeps_both_meanings():
    """Returning one of the two silently drops a constraint, and which one
    survives would depend on dict insertion order."""
    both = {"loyal": "three or more orders", "engaged": "two orders in 180 days"}

    got = lookup_definition(both, "engaged loyal customers")

    assert got is not None
    assert "three or more orders" in got
    assert "two orders in 180 days" in got


def test_a_reworded_multiword_term_beats_its_broader_word():
    """"share of loyal customers" must find `loyal share`, not fall through to
    plain `loyal` and lose the numerator/denominator rule the specific term
    carries."""
    both = {"loyal": "three or more orders", "loyal share": "the share rule"}

    assert lookup_definition(both, "share of loyal customers") == "the share rule"
