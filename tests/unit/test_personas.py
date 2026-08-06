import pytest

from retail_agent.store.personas import (
    DEFAULT_PERSONA,
    CachedPersonaStore,
    InMemoryPersonaStore,
    PersonaStore,
    active_body,
)
from tests.support.persona_store_contract import PersonaStoreContract


class TestInMemoryPersonaStore(PersonaStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryPersonaStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryPersonaStore(), PersonaStore)


# --- the TTL cache: an edit lands without a restart ---


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds


def test_repeated_reads_within_the_ttl_hit_the_store_once():
    inner = InMemoryPersonaStore()
    inner.seed(DEFAULT_PERSONA)
    reads = []
    original = inner.active
    inner.active = lambda: (reads.append(1), original())[1]

    cached = CachedPersonaStore(inner, ttl_seconds=60, now=Clock())
    for _ in range(5):
        cached.active()

    assert len(reads) == 1


def test_an_edit_lands_once_the_ttl_expires():
    """The requirement is "without redeployment", not "instantly"."""
    clock = Clock()
    inner = InMemoryPersonaStore()
    inner.seed(DEFAULT_PERSONA)
    cached = CachedPersonaStore(inner, ttl_seconds=60, now=clock)
    assert cached.active().name == DEFAULT_PERSONA.name

    inner.save(name="terse", body="Be brief.", updated_by="ceo")
    inner.activate(name="terse")
    assert cached.active().name == DEFAULT_PERSONA.name, "still cached"

    clock.seconds = 61
    assert cached.active().name == "terse"


def test_an_edit_through_the_cache_is_visible_immediately():
    """The person who just typed /persona activate should not wait a minute to
    see whether it worked."""
    cached = CachedPersonaStore(InMemoryPersonaStore(), ttl_seconds=60, now=Clock())
    cached.seed(DEFAULT_PERSONA)
    cached.active()

    cached.save(name="terse", body="Be brief.", updated_by="ceo")
    cached.activate(name="terse")

    assert cached.active().name == "terse"


# --- the slot always has something in it ---


def test_no_store_falls_back_to_the_built_in_persona():
    assert active_body(None) == DEFAULT_PERSONA.body


def test_nothing_active_falls_back_to_the_built_in_persona():
    assert active_body(InMemoryPersonaStore()) == DEFAULT_PERSONA.body


def test_an_empty_body_falls_back_rather_than_leaving_the_slot_blank():
    """An empty slot opens the prompt with the safety rules and no role, which
    changes how every answer reads for a reason nobody chose."""
    store = InMemoryPersonaStore()
    store.save(name="blank", body="   ", updated_by="ceo")
    store.activate(name="blank")

    assert active_body(store) == DEFAULT_PERSONA.body


def test_a_broken_store_falls_back_rather_than_failing_the_turn():
    class Broken:
        def active(self):
            raise RuntimeError("database gone")

    assert active_body(Broken()) == DEFAULT_PERSONA.body
