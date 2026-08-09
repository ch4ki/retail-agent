"""One contract, two implementations.

Personas are global — the CEO sets the house voice. Preferences are the
opposite: Manager A wants tables, Manager B wants bullets, and neither should
ever see the other's setting. Every method here is owner-scoped, and the tests
say so.
"""

from retail_agent.store.preferences import DEFAULT_PREFERENCES


class PreferenceStoreContract:
    """Subclass and provide a `store` fixture."""

    def test_an_unknown_user_gets_the_defaults(self, store):
        prefs = store.get(user_id="nobody")

        assert prefs.answer_format == DEFAULT_PREFERENCES.answer_format
        assert prefs.depth == DEFAULT_PREFERENCES.depth

    def test_a_set_preference_reads_back(self, store):
        store.set(user_id="dana", answer_format="bullets")

        assert store.get(user_id="dana").answer_format == "bullets"

    def test_preferences_do_not_leak_between_users(self, store):
        store.set(user_id="dana", answer_format="bullets")
        store.set(user_id="sam", answer_format="table")

        assert store.get(user_id="dana").answer_format == "bullets"
        assert store.get(user_id="sam").answer_format == "table"

    def test_setting_one_field_leaves_the_others_alone(self, store):
        store.set(user_id="dana", answer_format="bullets", depth="deep")

        store.set(user_id="dana", depth="summary")

        prefs = store.get(user_id="dana")
        assert prefs.depth == "summary"
        assert prefs.answer_format == "bullets", "not reset by an unrelated edit"

    def test_setting_nothing_changes_nothing(self, store):
        store.set(user_id="dana", answer_format="bullets")

        store.set(user_id="dana")

        assert store.get(user_id="dana").answer_format == "bullets"

    def test_max_table_rows_round_trips(self, store):
        store.set(user_id="dana", max_table_rows=5)

        assert store.get(user_id="dana").max_table_rows == 5

    def test_a_later_edit_wins(self, store):
        store.set(user_id="dana", depth="deep")
        store.set(user_id="dana", depth="summary")

        assert store.get(user_id="dana").depth == "summary"

    # --- the free-text notes list ---

    def test_an_unknown_user_has_no_notes(self, store):
        assert store.list_notes(user_id="nobody") == []

    def test_replaced_notes_read_back_in_order(self, store):
        store.replace_notes(user_id="dana", notes=["show prices in euros", "keep it short"])

        assert store.list_notes(user_id="dana") == [
            "show prices in euros",
            "keep it short",
        ]

    def test_notes_do_not_leak_between_users(self, store):
        store.replace_notes(user_id="dana", notes=["show prices in euros"])
        store.replace_notes(user_id="sam", notes=["always show the SQL"])

        assert store.list_notes(user_id="dana") == ["show prices in euros"]
        assert store.list_notes(user_id="sam") == ["always show the SQL"]

    def test_replacing_with_an_empty_list_clears_them(self, store):
        store.replace_notes(user_id="dana", notes=["show prices in euros"])

        store.replace_notes(user_id="dana", notes=[])

        assert store.list_notes(user_id="dana") == []
