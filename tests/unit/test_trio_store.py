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


# --- re-seeding a corpus that is already in the database ---


def _edited(trio, **fields):
    from dataclasses import replace

    return replace(trio, **fields)


def test_a_corpus_that_matches_the_seed_reports_no_drift():
    from retail_agent.knowledge.trios import seed_drift

    store = InMemoryTrioStore(SEED_TRIOS)

    assert seed_drift(store, SEED_TRIOS) == {}


def test_a_trio_the_store_has_never_seen_is_missing():
    from retail_agent.knowledge.trios import seed_drift

    store = InMemoryTrioStore(SEED_TRIOS[1:])

    assert seed_drift(store, SEED_TRIOS) == {SEED_TRIOS[0].id: "missing"}


def test_a_definition_changed_since_seeding_is_drift():
    """The case this exists for. `seed` inserts what is absent and leaves what
    is there, so an edit to `seeds.py` never reaches a database that already
    ran once — the agent keeps answering from the corpus it was first given,
    and nothing says so."""
    from retail_agent.knowledge.trios import seed_drift

    stale = _edited(SEED_TRIOS[0], metric_definitions={"churn": "something older"})
    store = InMemoryTrioStore([stale, *SEED_TRIOS[1:]])

    assert seed_drift(store, SEED_TRIOS) == {SEED_TRIOS[0].id: "changed"}


def test_reseeding_writes_only_what_drifted():
    from retail_agent.knowledge.trios import reseed, seed_drift

    stale = _edited(SEED_TRIOS[0], metric_definitions={"churn": "something older"})
    store = InMemoryTrioStore([stale, *SEED_TRIOS[1:]])

    written = reseed(store, SEED_TRIOS)

    assert written == [SEED_TRIOS[0].id]
    assert seed_drift(store, SEED_TRIOS) == {}


def test_reseeding_a_matching_corpus_writes_nothing():
    from retail_agent.knowledge.trios import reseed

    assert reseed(InMemoryTrioStore(SEED_TRIOS), SEED_TRIOS) == []
