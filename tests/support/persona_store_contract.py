"""One contract, two implementations.

Unlike reports, personas are not owned by anyone: the CEO changes the tone for
everybody. So there is no owner predicate here — and that asymmetry is the point
of having the contract state it explicitly.
"""

import pytest

from retail_agent.store.personas import DEFAULT_PERSONA


class PersonaStoreContract:
    """Subclass and provide a `store` fixture."""

    def test_a_new_name_starts_at_version_one(self, store):
        persona = store.save(name="terse", body="Be brief.", updated_by="ceo")

        assert persona.version == 1
        assert persona.name == "terse"

    def test_saving_the_same_name_again_creates_a_new_version(self, store):
        store.save(name="terse", body="Be brief.", updated_by="ceo")
        second = store.save(name="terse", body="Be very brief.", updated_by="ceo")

        assert second.version == 2

    def test_editing_never_destroys_the_previous_version(self, store):
        """Rollback is only possible if the old body still exists."""
        store.save(name="terse", body="Be brief.", updated_by="ceo")
        store.save(name="terse", body="Be very brief.", updated_by="ceo")

        assert store.get(name="terse", version=1).body == "Be brief."

    def test_nothing_is_active_until_something_is_activated(self, store):
        store.save(name="terse", body="Be brief.", updated_by="ceo")

        assert store.active() is None

    def test_activating_makes_it_the_active_persona(self, store):
        store.save(name="terse", body="Be brief.", updated_by="ceo")

        store.activate(name="terse")

        active = store.active()
        assert active.name == "terse"
        assert active.body == "Be brief."

    def test_only_one_persona_is_active_at_a_time(self, store):
        store.save(name="terse", body="Be brief.", updated_by="ceo")
        store.save(name="warm", body="Be friendly.", updated_by="ceo")
        store.activate(name="terse")

        store.activate(name="warm")

        assert store.active().name == "warm"
        assert len([p for p in store.list_personas() if p.is_active]) == 1

    def test_activating_defaults_to_the_latest_version(self, store):
        store.save(name="terse", body="v1", updated_by="ceo")
        store.save(name="terse", body="v2", updated_by="ceo")

        store.activate(name="terse")

        assert store.active().version == 2

    def test_activating_an_older_version_is_a_rollback(self, store):
        store.save(name="terse", body="v1", updated_by="ceo")
        store.save(name="terse", body="v2 went badly", updated_by="ceo")
        store.activate(name="terse")

        store.activate(name="terse", version=1)

        assert store.active().body == "v1"
        assert store.active().version == 1

    def test_activating_an_unknown_name_changes_nothing(self, store):
        store.save(name="terse", body="Be brief.", updated_by="ceo")
        store.activate(name="terse")

        with pytest.raises(KeyError):
            store.activate(name="nope")

        assert store.active().name == "terse"

    def test_list_shows_one_entry_per_name_at_its_latest_version(self, store):
        store.save(name="terse", body="v1", updated_by="ceo")
        store.save(name="terse", body="v2", updated_by="ceo")
        store.save(name="warm", body="v1", updated_by="ceo")

        listed = {p.name: p.version for p in store.list_personas()}

        assert listed == {"terse": 2, "warm": 1}

    def test_the_editor_is_recorded(self, store):
        """A tone change is a production change; who made it has to be on the
        record."""
        persona = store.save(name="terse", body="Be brief.", updated_by="dana")

        assert persona.updated_by == "dana"

    def test_seeding_is_idempotent(self, store):
        """`migrate` runs more than once. Seeding must not pile up versions of
        an unchanged default."""
        store.seed(DEFAULT_PERSONA)
        store.seed(DEFAULT_PERSONA)

        matching = [p for p in store.list_personas() if p.name == DEFAULT_PERSONA.name]
        assert len(matching) == 1
        assert matching[0].version == 1

    def test_seeding_activates_the_default_when_nothing_is_active(self, store):
        store.seed(DEFAULT_PERSONA)

        assert store.active().name == DEFAULT_PERSONA.name

    def test_seeding_does_not_override_a_chosen_persona(self, store):
        """Restarting the app must not silently undo the CEO's choice."""
        store.save(name="terse", body="Be brief.", updated_by="ceo")
        store.activate(name="terse")

        store.seed(DEFAULT_PERSONA)

        assert store.active().name == "terse"
