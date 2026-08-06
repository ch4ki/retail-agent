import pytest

from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.knowledge.trios import InMemoryTrioStore, TrioStore, live_trios
from tests.support.trio_store_contract import TrioStoreContract, trio


class TestInMemoryTrioStore(TrioStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryTrioStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryTrioStore(), TrioStore)


def test_live_trios_accepts_a_store():
    assert len(live_trios(InMemoryTrioStore(SEED_TRIOS))) == len(SEED_TRIOS)


def test_live_trios_accepts_a_plain_list():
    assert len(live_trios(list(SEED_TRIOS))) == len(SEED_TRIOS)


def test_live_trios_filters_superseded_from_a_plain_list():
    old = trio("old", superseded_by="new")

    assert live_trios([old, trio("new")]) == [trio("new")]


def test_a_broken_store_costs_grounding_not_the_answer():
    class Broken:
        def live(self):
            raise RuntimeError("database gone")

    assert live_trios(Broken()) == []


def test_no_corpus_is_a_valid_state():
    assert live_trios(None) == []
