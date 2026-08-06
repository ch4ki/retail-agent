"""One contract, two implementations.

Per user, like preferences and unlike personas: one manager's working
definition of "loyal" must not silently redefine the term for everyone else.
"""


class DefinitionStoreContract:
    """Subclass and provide a `store` fixture."""

    def test_a_definition_reads_back(self, store):
        store.remember(user_id="dana", term="loyal", definition="3+ orders")

        assert store.lookup(user_id="dana", term="loyal").definition == "3+ orders"

    def test_terms_are_matched_case_insensitively(self, store):
        store.remember(user_id="dana", term="Loyal", definition="3+ orders")

        assert store.lookup(user_id="dana", term="LOYAL") is not None

    def test_definitions_do_not_leak_between_users(self, store):
        store.remember(user_id="dana", term="loyal", definition="3+ orders")

        assert store.lookup(user_id="sam", term="loyal") is None

    def test_an_unknown_term_returns_nothing(self, store):
        assert store.lookup(user_id="dana", term="nonsense") is None

    def test_redefining_replaces_rather_than_duplicates(self, store):
        store.remember(user_id="dana", term="loyal", definition="3+ orders")
        store.remember(user_id="dana", term="loyal", definition="5+ orders")

        assert store.lookup(user_id="dana", term="loyal").definition == "5+ orders"
        assert len(store.list_definitions(user_id="dana")) == 1

    def test_listing_is_scoped_and_sorted(self, store):
        store.remember(user_id="dana", term="loyal", definition="a")
        store.remember(user_id="dana", term="at risk", definition="b")
        store.remember(user_id="sam", term="churn", definition="c")

        assert [d.term for d in store.list_definitions(user_id="dana")] == [
            "at risk",
            "loyal",
        ]

    def test_forgetting_a_definition(self, store):
        store.remember(user_id="dana", term="loyal", definition="3+ orders")

        assert store.forget(user_id="dana", term="loyal") is True
        assert store.lookup(user_id="dana", term="loyal") is None

    def test_forgetting_something_absent_reports_it(self, store):
        assert store.forget(user_id="dana", term="loyal") is False

    def test_a_very_long_definition_is_truncated(self, store):
        from retail_agent.store.definitions import MAX_DEFINITION_CHARS

        entry = store.remember(user_id="dana", term="loyal", definition="x" * 5_000)

        assert len(entry.definition) <= MAX_DEFINITION_CHARS
