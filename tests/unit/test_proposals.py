"""Candidate definitions offered when the corpus settles nothing.

The options are a convenience on top of a question that gets asked either way,
so the tests that matter most here are the ones about failing quietly.
"""

from tests.component.conftest import ScriptedLLM

from retail_agent.knowledge.proposals import propose

SCHEMA = "orders(id, user_id, created_at)\norder_items(id, sale_price)"


def _deps_for(llm):
    """`propose` takes deps now, so it can retry and fall over the way every
    other model call does."""
    from types import SimpleNamespace

    from retail_agent.config import Settings

    return SimpleNamespace(
        llm=llm, llm_fallbacks=[], settings=Settings(_env_file=None)
    )


def suggest(replies, **overrides):
    llm = ScriptedLLM(replies)
    kwargs = dict(
        question="who are my loyal customers?",
        term="loyal",
        schema=SCHEMA,
    )
    kwargs.update(overrides)
    return propose(_deps_for(llm), **kwargs), llm


def test_the_proposed_definitions_come_back_in_order():
    options, _ = suggest(
        [{"definitions": ["3 or more orders, ever", "2 orders in 12 months"]}]
    )

    assert options == ["3 or more orders, ever", "2 orders in 12 months"]


def test_a_model_failure_costs_the_options_and_not_the_turn():
    """The user is still asked; they just type their own. Same bargain `recall`
    makes — retrieval is an improvement, never a dependency."""

    class Broken(ScriptedLLM):
        def with_structured_output(self, schema, **kwargs):
            raise RuntimeError("the provider is down")

    assert propose(
        _deps_for(Broken([])), question="q", term="loyal", schema=SCHEMA
    ) == []


def test_a_reply_the_schema_rejects_yields_no_options():
    options, _ = suggest([{"wrong_field": ["nope"]}])

    assert options == []


def test_the_model_is_told_the_term_and_the_question():
    """Without the question the options describe the term in the abstract, and
    the point is to fit what was actually asked.

    No hint any more: the words come from the executive rather than from a dict
    that shipped a gloss with each one, so the model works out what has to be
    decided from the term and the schema."""
    _, llm = suggest([{"definitions": ["3 or more orders"]}])

    prompt = llm.prompts[0]
    assert "loyal" in prompt
    assert "who are my loyal customers?" in prompt


def test_the_model_is_shown_the_schema_so_an_option_cannot_invent_a_column():
    _, llm = suggest([{"definitions": ["3 or more orders"]}])

    assert "order_items" in llm.prompts[0]


def test_terms_already_settled_this_turn_are_shown():
    """The second prompt of a turn should read the first answer, or it will
    offer a definition of `top` that contradicts the agreed `loyal`."""
    _, llm = suggest(
        [{"definitions": ["the 10 highest by revenue"]}],
        term="top",
        settled={"loyal": "2 or more orders in the last 12 months"},
    )

    assert "2 or more orders in the last 12 months" in llm.prompts[0]


def test_blank_and_duplicate_options_are_dropped():
    options, _ = suggest(
        [{"definitions": ["3 or more orders", "  ", "3 or more orders", "2 in a year"]}]
    )

    assert options == ["3 or more orders", "2 in a year"]


def test_no_more_than_four_options_are_offered():
    """A numbered list someone has to read before choosing. Past four it stops
    being a choice and becomes a wall."""
    options, _ = suggest([{"definitions": [f"rule {n}" for n in range(9)]}])

    assert len(options) == 4


def test_an_over_long_option_is_cut_to_what_the_store_will_keep():
    """Offering something longer than `remember` will save would show the user
    one definition and record another."""
    from retail_agent.store.definitions import MAX_DEFINITION_CHARS

    options, _ = suggest([{"definitions": ["x" * (MAX_DEFINITION_CHARS + 50)]}])

    assert len(options[0]) == MAX_DEFINITION_CHARS
