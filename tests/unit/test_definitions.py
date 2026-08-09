import pytest

from retail_agent.store.definitions import (
    DefinitionStore,
    InMemoryDefinitionStore,
    all_definitions,
    personal_definitions_block,
)
from tests.support.definition_store_contract import DefinitionStoreContract


class TestInMemoryDefinitionStore(DefinitionStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryDefinitionStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryDefinitionStore(), DefinitionStore)


def test_all_definitions_returns_everything_this_user_gave():
    """Everything, not a lookup of named terms: nothing computes a list of
    terms to look up any more, so the analyst asks for the lot."""
    store = InMemoryDefinitionStore()
    store.remember(user_id="dana", term="loyal", definition="3+ orders")
    store.remember(user_id="dana", term="LGB", definition="low gross basket")
    store.remember(user_id="sam", term="top", definition="by revenue")

    found = all_definitions(store, "dana")

    assert found == {"loyal": "3+ orders", "lgb": "low gross basket"}


def test_one_users_definitions_do_not_reach_another():
    store = InMemoryDefinitionStore()
    store.remember(user_id="sam", term="top", definition="by revenue")

    assert all_definitions(store, "dana") == {}


def test_a_broken_store_does_not_fail_the_turn():
    """Losing the store costs a question the user already answered, not the
    answer."""

    class Broken:
        def list_definitions(self, *, user_id):
            raise RuntimeError("database gone")

    assert all_definitions(Broken(), "dana") == {}


def test_no_store_means_no_definitions():
    assert all_definitions(None, "dana") == {}


def test_personal_definitions_are_labelled_as_the_users_own():
    """A working definition one manager typed must not be presented as though
    the analytics team had agreed it."""
    block = personal_definitions_block({"loyal": "3+ orders"})

    assert "user's own" in block
    assert "3+ orders" in block

