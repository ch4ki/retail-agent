import pytest

from retail_agent.store.preferences import (
    DEFAULT_PREFERENCES,
    InMemoryPreferenceStore,
    PreferenceError,
    PreferenceStore,
    coerce,
    preferred,
)
from tests.support.preference_store_contract import PreferenceStoreContract


class TestInMemoryPreferenceStore(PreferenceStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryPreferenceStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryPreferenceStore(), PreferenceStore)


# --- validation happens before anything is stored ---


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("show_attempt_footnote", "yes"),
        ("colour", "blue"),
    ],
)
def test_a_bad_value_is_rejected_with_a_usable_message(field, value):
    """A typo should be a message, not a preference silently set to nonsense."""
    with pytest.raises(PreferenceError) as excinfo:
        coerce(field, value)

    assert str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("show_attempt_footnote", "false", False),
        ("show_attempt_footnote", "TRUE", True),
    ],
)
def test_good_values_are_parsed(field, value, expected):
    assert coerce(field, value) == expected


@pytest.mark.parametrize("field", ["answer_format", "depth", "max_table_rows"])
def test_a_retired_setting_is_refused_rather_than_silently_stored(field):
    """These were stored, validated, and read by nothing — a setting that
    silently does nothing is worse than one never offered. `/prefs` must now
    say the setting does not exist instead of accepting it."""
    with pytest.raises(PreferenceError, match="unknown"):
        coerce(field, "anything")


# --- never fail a turn over a layout setting ---


def test_no_store_yields_the_defaults():
    assert preferred(None, "dana") == DEFAULT_PREFERENCES


def test_a_broken_store_yields_the_defaults():
    class Broken:
        def get(self, *, user_id):
            raise RuntimeError("database gone")

    assert preferred(Broken(), "dana") == DEFAULT_PREFERENCES


# --- the free-text notes list ---


def _notes_store():
    return InMemoryPreferenceStore()


def test_a_note_is_saved_and_reads_back():
    from retail_agent.store.preferences import add_note, notes_for

    store = _notes_store()

    assert add_note(store, user_id="dana", note="show prices in euros") == "added"
    assert notes_for(store, "dana") == ["show prices in euros"]


def test_notes_keep_the_order_they_were_added_in():
    from retail_agent.store.preferences import add_note, notes_for

    store = _notes_store()
    add_note(store, user_id="dana", note="show prices in euros")
    add_note(store, user_id="dana", note="keep answers under three sentences")

    assert notes_for(store, "dana") == [
        "show prices in euros",
        "keep answers under three sentences",
    ]


def test_the_same_note_again_is_a_duplicate_not_a_second_row():
    """Said twice, in different words on the page but the same words in
    substance — one note, not two lines of the same request in the prompt."""
    from retail_agent.store.preferences import add_note, notes_for

    store = _notes_store()
    add_note(store, user_id="dana", note="show prices in euros")

    assert add_note(store, user_id="dana", note="  Show Prices In Euros ") == "duplicate"
    assert notes_for(store, "dana") == ["show prices in euros"]


def test_a_note_over_the_length_limit_is_rejected_rather_than_truncated():
    """Truncating would record a preference the user did not write."""
    from retail_agent.store.preferences import MAX_NOTE_CHARS, add_note, notes_for

    store = _notes_store()

    outcome = add_note(store, user_id="dana", note="x" * (MAX_NOTE_CHARS + 1))

    assert outcome == "too_long"
    assert notes_for(store, "dana") == []


def test_a_note_exactly_at_the_limit_is_accepted():
    from retail_agent.store.preferences import MAX_NOTE_CHARS, add_note

    store = _notes_store()

    assert add_note(store, user_id="dana", note="x" * MAX_NOTE_CHARS) == "added"


def test_the_cap_stops_the_prompt_block_growing_without_end():
    from retail_agent.store.preferences import MAX_NOTES, add_note, notes_for

    store = _notes_store()
    for index in range(MAX_NOTES):
        assert add_note(store, user_id="dana", note=f"preference {index}") == "added"

    assert add_note(store, user_id="dana", note="one more") == "full"
    assert len(notes_for(store, "dana")) == MAX_NOTES


def test_an_empty_note_records_nothing():
    from retail_agent.store.preferences import add_note, notes_for

    store = _notes_store()

    assert add_note(store, user_id="dana", note="   ") == "empty"
    assert notes_for(store, "dana") == []


def test_a_removed_note_is_gone_and_the_rest_survive():
    from retail_agent.store.preferences import add_note, notes_for, remove_note

    store = _notes_store()
    add_note(store, user_id="dana", note="show prices in euros")
    add_note(store, user_id="dana", note="keep it short")

    assert remove_note(store, user_id="dana", note="  SHOW prices in euros ") is True
    assert notes_for(store, "dana") == ["keep it short"]


def test_removing_something_never_saved_changes_nothing():
    from retail_agent.store.preferences import add_note, notes_for, remove_note

    store = _notes_store()
    add_note(store, user_id="dana", note="keep it short")

    assert remove_note(store, user_id="dana", note="show prices in euros") is False
    assert notes_for(store, "dana") == ["keep it short"]


def test_notes_do_not_leak_between_users_through_the_helpers():
    from retail_agent.store.preferences import add_note, notes_for

    store = _notes_store()
    add_note(store, user_id="dana", note="show prices in euros")

    assert notes_for(store, "sam") == []


def test_no_store_yields_no_notes():
    from retail_agent.store.preferences import notes_for

    assert notes_for(None, "dana") == []


def test_a_broken_store_costs_the_notes_not_the_turn():
    from retail_agent.store.preferences import notes_for

    class Broken:
        def list_notes(self, *, user_id):
            raise RuntimeError("database gone")

    assert notes_for(Broken(), "dana") == []


def test_an_empty_list_contributes_nothing_to_the_prompt():
    """A heading with no notes under it is noise the model has to read past."""
    from retail_agent.store.preferences import preference_block

    assert preference_block([]) == ""


def test_the_block_names_every_note():
    from retail_agent.store.preferences import preference_block

    block = preference_block(["show prices in euros", "keep it short"])

    assert "show prices in euros" in block
    assert "keep it short" in block
    assert block.count("\n- ") == 2, "one bullet per note"
