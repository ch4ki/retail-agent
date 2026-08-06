import pytest

from retail_agent.store.preferences import (
    DEFAULT_PREFERENCES,
    InMemoryPreferenceStore,
    PreferenceError,
    PreferenceStore,
    Preferences,
    coerce,
    preferred,
    style_instruction,
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
        ("answer_format", "tables"),      # near miss
        ("depth", "verbose"),
        ("max_table_rows", "lots"),
        ("max_table_rows", "0"),
        ("max_table_rows", "1000"),
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
        ("answer_format", "bullets", "bullets"),
        ("depth", "deep", "deep"),
        ("max_table_rows", "5", 5),
        ("show_attempt_footnote", "false", False),
        ("show_attempt_footnote", "TRUE", True),
    ],
)
def test_good_values_are_parsed(field, value, expected):
    assert coerce(field, value) == expected


# --- never fail a turn over a layout setting ---


def test_no_store_yields_the_defaults():
    assert preferred(None, "dana") == DEFAULT_PREFERENCES


def test_a_broken_store_yields_the_defaults():
    class Broken:
        def get(self, *, user_id):
            raise RuntimeError("database gone")

    assert preferred(Broken(), "dana") == DEFAULT_PREFERENCES


# --- the prompt-side half ---


def test_each_format_produces_distinct_guidance():
    said = {
        style_instruction(Preferences(answer_format=fmt))
        for fmt in ("table", "bullets", "prose")
    }

    assert len(said) == 3, "the setting has to actually change the instruction"


def test_bullets_asks_against_tables():
    instruction = style_instruction(Preferences(answer_format="bullets"))

    assert "bullet" in instruction.lower()
    assert "table" in instruction.lower(), "and says what not to do"


def test_depth_changes_how_much_is_asked_for():
    summary = style_instruction(Preferences(depth="summary"))
    deep = style_instruction(Preferences(depth="deep"))

    assert "Stop there" in summary
    assert len(deep) > len(summary)
