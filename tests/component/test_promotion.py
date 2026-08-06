"""Promoting a personal definition into the shared corpus.

§5.1's rule: nothing merges automatically. An agent that writes its own ground
truth drifts, and a poisoned corpus is expensive to recover from. So promotion
is an explicit human act — here, a command someone types — and it produces a
trio attributed to whoever promoted it.
"""

import pytest

from retail_agent.knowledge.promotion import PromotionError, promote_definition
from retail_agent.knowledge.trios import InMemoryTrioStore, live_trios
from retail_agent.store.definitions import InMemoryDefinitionStore


@pytest.fixture
def stores():
    definitions = InMemoryDefinitionStore()
    definitions.remember(
        user_id="dana", term="at risk", definition="no completed order in 120 days"
    )
    return definitions, InMemoryTrioStore()


def test_a_personal_definition_becomes_a_trio(stores):
    definitions, trios = stores

    trio = promote_definition(
        definitions, trios, user_id="dana", term="at risk", promoted_by="dana"
    )

    assert trio.metric_definitions == {"at risk": "no completed order in 120 days"}
    assert trio.id in {t.id for t in live_trios(trios)}


def test_the_promoted_trio_records_who_promoted_it(stores):
    """A shared definition is a production change; it needs an author."""
    definitions, trios = stores

    trio = promote_definition(
        definitions, trios, user_id="dana", term="at risk", promoted_by="dana"
    )

    assert trio.author == "dana"
    assert trio.approved_at is not None


def test_the_term_becomes_answerable_for_everyone(stores):
    """The point of promoting: another user stops being asked."""
    definitions, trios = stores
    promote_definition(
        definitions, trios, user_id="dana", term="at risk", promoted_by="dana"
    )

    from retail_agent.knowledge.trios import unresolved

    assert unresolved("which customers are at risk?", live_trios(trios)) == []


def test_promoting_something_never_defined_is_refused(stores):
    definitions, trios = stores

    with pytest.raises(PromotionError):
        promote_definition(
            definitions, trios, user_id="dana", term="nonsense", promoted_by="dana"
        )


def test_promoting_a_term_the_corpus_already_defines_supersedes_it(stores):
    """Definitions change. The old trio stays readable so a report written
    under it can still be explained."""
    definitions, trios = stores
    first = promote_definition(
        definitions, trios, user_id="dana", term="at risk", promoted_by="dana"
    )

    definitions.remember(user_id="dana", term="at risk", definition="no order in 60 days")
    second = promote_definition(
        definitions, trios, user_id="dana", term="at risk", promoted_by="sam"
    )

    assert {t.id for t in live_trios(trios)} == {second.id}
    superseded = trios.get(first.id)
    assert superseded.superseded_by == second.id
    assert superseded.metric_definitions == {"at risk": "no completed order in 120 days"}


def test_a_promoted_trio_carries_the_question_that_prompted_it(stores):
    """Retrieval matches on the question text, so a trio with no question is
    findable only by its tags."""
    definitions, trios = stores

    trio = promote_definition(
        definitions,
        trios,
        user_id="dana",
        term="at risk",
        promoted_by="dana",
        question="which customers are at risk?",
    )

    assert "at risk" in trio.question
    assert "at risk" in trio.tags
