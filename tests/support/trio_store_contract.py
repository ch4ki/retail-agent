"""One contract, two implementations.

The corpus is the system's ground truth, so the properties that matter are
about history: editing a definition must never destroy the one that produced
last quarter's report, and a superseded trio must never answer a new question.
"""

import pytest

from retail_agent.knowledge.trios import Trio


def trio(id="churn-90", **overrides):
    base = dict(
        id=id,
        question="Which customers churned?",
        sql="SELECT 1",
        report="Churn rose to 4.1%.",
        metric_definitions={"churn": "no order in 90 days"},
        tags=("churn", "retention"),
        author="analytics",
    )
    base.update(overrides)
    return Trio(**base)


class TrioStoreContract:
    """Subclass and provide a `store` fixture."""

    def test_an_added_trio_reads_back(self, store):
        store.add(trio())

        found = store.get("churn-90")
        assert found.question == "Which customers churned?"
        assert found.metric_definitions == {"churn": "no order in 90 days"}

    def test_tags_survive_the_round_trip(self, store):
        store.add(trio())

        assert set(store.get("churn-90").tags) == {"churn", "retention"}

    def test_an_unknown_id_returns_nothing(self, store):
        assert store.get("nope") is None

    def test_live_returns_everything_not_superseded(self, store):
        store.add(trio("a"))
        store.add(trio("b"))

        assert {t.id for t in store.live()} == {"a", "b"}

    def test_superseding_hides_the_old_trio_from_new_questions(self, store):
        store.add(trio("old", metric_definitions={"churn": "60 days"}))
        store.add(trio("new", metric_definitions={"churn": "90 days"}))

        store.supersede(old_id="old", new_id="new")

        assert {t.id for t in store.live()} == {"new"}

    def test_a_superseded_trio_is_still_readable_by_id(self, store):
        """A report written last quarter has to remain explicable against the
        definition that produced it."""
        store.add(trio("old", metric_definitions={"churn": "60 days"}))
        store.add(trio("new"))
        store.supersede(old_id="old", new_id="new")

        old = store.get("old")
        assert old is not None
        assert old.superseded_by == "new"
        assert old.metric_definitions == {"churn": "60 days"}

    def test_superseding_an_unknown_trio_is_an_error(self, store):
        store.add(trio("new"))

        with pytest.raises(KeyError):
            store.supersede(old_id="nope", new_id="new")

    def test_adding_the_same_id_twice_replaces_it(self, store):
        store.add(trio("a", report="first"))
        store.add(trio("a", report="second"))

        assert store.get("a").report == "second"
        assert len(store.live()) == 1

    def test_seeding_is_idempotent(self, store):
        """`migrate` runs more than once, and a duplicated corpus would double
        every definition in the prompt."""
        store.seed([trio("a"), trio("b")])
        store.seed([trio("a"), trio("b")])

        assert len(store.live()) == 2

    def test_seeding_does_not_overwrite_an_edited_trio(self, store):
        """An analyst's edit must survive a restart."""
        store.seed([trio("a", report="original")])
        store.add(trio("a", report="edited by an analyst"))

        store.seed([trio("a", report="original")])

        assert store.get("a").report == "edited by an analyst"
