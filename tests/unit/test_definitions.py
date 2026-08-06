import pytest

from retail_agent.store.definitions import (
    DefinitionStore,
    InMemoryDefinitionStore,
    ask_for_definition,
    personal_definitions_block,
    remembered,
)
from tests.support.definition_store_contract import DefinitionStoreContract


class TestInMemoryDefinitionStore(DefinitionStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryDefinitionStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryDefinitionStore(), DefinitionStore)


def test_remembered_returns_only_what_is_known():
    store = InMemoryDefinitionStore()
    store.remember(user_id="dana", term="loyal", definition="3+ orders")

    found = remembered(store, "dana", ["loyal", "at risk"])

    assert found == {"loyal": "3+ orders"}


def test_a_broken_store_does_not_fail_the_turn():
    """Losing the store costs a question the user already answered, not the
    answer."""

    class Broken:
        def lookup(self, *, user_id, term):
            raise RuntimeError("database gone")

    assert remembered(Broken(), "dana", ["loyal"]) == {}


def test_personal_definitions_are_labelled_as_the_users_own():
    """A working definition one manager typed must not be presented as though
    the analytics team had agreed it."""
    block = personal_definitions_block({"loyal": "3+ orders"})

    assert "user's own" in block
    assert "3+ orders" in block


def test_the_question_names_the_term_and_offers_an_example():
    question = ask_for_definition("loyal", "what makes a customer loyal")

    assert "loyal" in question
    assert "For example" in question
    assert "enter" in question.lower(), "opting out has to be visible"
