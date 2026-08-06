"""One contract, two implementations. Anything asserted here must hold for
Postgres and for the in-memory store, or component tests are testing a fiction.
"""


class ReportStoreContract:
    """Subclass and provide a `store` fixture."""

    def test_save_returns_a_report_with_an_id(self, store):
        report = store.save(
            owner_id="dana", session_id="s1", title="Q1", body="revenue rose"
        )
        assert report.id
        assert report.title == "Q1"
        assert report.deleted_at is None

    def test_get_is_scoped_to_the_owner(self, store):
        report = store.save(
            owner_id="dana", session_id="s1", title="Q1", body="secret"
        )
        assert store.get(owner_id="dana", report_id=report.id) is not None
        assert store.get(owner_id="sam", report_id=report.id) is None

    def test_list_excludes_other_owners(self, store):
        store.save(owner_id="dana", session_id="s1", title="A", body="x")
        store.save(owner_id="sam", session_id="s1", title="B", body="y")

        titles = [r.title for r in store.list_reports(owner_id="dana")]

        assert titles == ["A"]

    def test_resolve_by_term_searches_title_and_body(self, store):
        store.save(
            owner_id="dana",
            session_id="s1",
            title="Q1 review",
            body="Calvin Klein outperformed the category",
        )
        store.save(owner_id="dana", session_id="s1", title="Q2 plan", body="Levi")

        found = store.resolve(owner_id="dana", term="Calvin Klein")

        assert [r.title for r in found] == ["Q1 review"]

    def test_resolve_by_session_uses_the_session_id(self, store):
        store.save(owner_id="dana", session_id="s1", title="A", body="x")
        store.save(owner_id="dana", session_id="s2", title="B", body="x")

        found = store.resolve(owner_id="dana", session_id="s1")

        assert [r.title for r in found] == ["A"]

    def test_resolve_never_crosses_owners(self, store):
        store.save(owner_id="sam", session_id="s1", title="A", body="Calvin Klein")

        assert store.resolve(owner_id="dana", term="Calvin Klein") == []

    def test_soft_delete_tombstones_rather_than_removing(self, store):
        report = store.save(owner_id="dana", session_id="s1", title="A", body="x")

        count = store.soft_delete(
            owner_id="dana", report_ids=[report.id], action_id="a1", token="y"
        )

        assert count == 1
        assert store.list_reports(owner_id="dana") == []
        assert store.get(owner_id="dana", report_id=report.id).deleted_at is not None

    def test_replaying_an_action_id_deletes_nothing_further(self, store):
        """A durable checkpointer and Studio's time travel both make a resume
        replayable. Without this the second one deletes a second time."""
        a = store.save(owner_id="dana", session_id="s1", title="A", body="x")
        b = store.save(owner_id="dana", session_id="s1", title="B", body="x")

        first = store.soft_delete(
            owner_id="dana", report_ids=[a.id], action_id="a1", token="y"
        )
        second = store.soft_delete(
            owner_id="dana", report_ids=[b.id], action_id="a1", token="y"
        )

        assert first == 1
        assert second == 0
        assert [r.title for r in store.list_reports(owner_id="dana")] == ["B"]

    def test_soft_delete_cannot_touch_another_owners_report(self, store):
        report = store.save(owner_id="sam", session_id="s1", title="A", body="x")

        count = store.soft_delete(
            owner_id="dana", report_ids=[report.id], action_id="a1", token="y"
        )

        assert count == 0
        assert len(store.list_reports(owner_id="sam")) == 1

    def test_undo_restores_the_last_action(self, store):
        report = store.save(owner_id="dana", session_id="s1", title="A", body="x")
        store.soft_delete(
            owner_id="dana", report_ids=[report.id], action_id="a1", token="y"
        )

        restored = store.undo(owner_id="dana")

        assert restored == 1
        assert [r.title for r in store.list_reports(owner_id="dana")] == ["A"]

    def test_undo_is_not_repeatable(self, store):
        report = store.save(owner_id="dana", session_id="s1", title="A", body="x")
        store.soft_delete(
            owner_id="dana", report_ids=[report.id], action_id="a1", token="y"
        )
        store.undo(owner_id="dana")

        assert store.undo(owner_id="dana") == 0

    def test_last_action_reports_what_was_deleted(self, store):
        report = store.save(owner_id="dana", session_id="s1", title="A", body="x")
        store.soft_delete(
            owner_id="dana", report_ids=[report.id], action_id="a1", token="DELETE 1"
        )

        entry = store.last_action(owner_id="dana")

        assert entry.report_ids == (report.id,)
        assert entry.token == "DELETE 1"
