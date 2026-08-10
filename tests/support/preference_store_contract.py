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

        assert (
            prefs.show_attempt_footnote == DEFAULT_PREFERENCES.show_attempt_footnote
        )

    def test_a_set_preference_reads_back(self, store):
        store.set(user_id="dana", show_attempt_footnote=False)

        assert store.get(user_id="dana").show_attempt_footnote is False

    def test_preferences_do_not_leak_between_users(self, store):
        store.set(user_id="dana", show_attempt_footnote=False)
        store.set(user_id="sam", show_attempt_footnote=True)

        assert store.get(user_id="dana").show_attempt_footnote is False
        assert store.get(user_id="sam").show_attempt_footnote is True

    def test_setting_nothing_changes_nothing(self, store):
        store.set(user_id="dana", show_attempt_footnote=False)

        store.set(user_id="dana")

        assert store.get(user_id="dana").show_attempt_footnote is False

    def test_a_later_edit_wins(self, store):
        store.set(user_id="dana", show_attempt_footnote=False)
        store.set(user_id="dana", show_attempt_footnote=True)

        assert store.get(user_id="dana").show_attempt_footnote is True

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
