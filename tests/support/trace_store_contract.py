"""One contract, two implementations — same discipline as the report store.

Metrics are computed from whatever this returns, so a drift between the
in-memory double and Postgres would show up as a wrong number on a dashboard
rather than as a failing test.
"""

from retail_agent.obs.traces import TraceRecord


def _record(turn_id="t1", **overrides):
    base = dict(
        turn_id=turn_id,
        session_id="s1",
        owner_id="dana",
        question="revenue in March?",
        intent="analyze",
        status="ok",
        answer="Revenue was $1.2M.",
        redactions=0,
        bytes_billed=2048,
        duration_ms=900,
        events=[("route", 100, "intent=analyze"), ("plan", 300, "1 step(s)")],
        attempts=[
            {
                "step_id": "step_1",
                "sql": "SELECT 1",
                "executed_sql": "SELECT 1 LIMIT 500",
                "violations": [],
                "error": None,
                "row_count": 3,
                "bytes_billed": 2048,
            }
        ],
    )
    base.update(overrides)
    return TraceRecord(**base)


class TraceStoreContract:
    """Subclass and provide a `store` fixture."""

    def test_a_saved_trace_can_be_read_back_by_turn_id(self, store):
        store.record(_record("abc123"))

        found = store.get(owner_id="dana", turn_id="abc123")

        assert found is not None
        assert found.question == "revenue in March?"
        assert found.status == "ok"

    def test_events_survive_the_round_trip_in_order(self, store):
        store.record(
            _record("abc123", events=[("route", 1, "a"), ("plan", 2, "b"), ("execute", 3, "c")])
        )

        found = store.get(owner_id="dana", turn_id="abc123")

        assert [e[0] for e in found.events] == ["route", "plan", "execute"]
        assert found.events[2][1] == 3

    def test_attempts_survive_the_round_trip(self, store):
        store.record(_record("abc123"))

        found = store.get(owner_id="dana", turn_id="abc123")

        assert found.attempts[0]["step_id"] == "step_1"
        assert found.attempts[0]["row_count"] == 3

    def test_the_reasons_behind_the_answer_survive_the_round_trip(self, store):
        """Which definitions were consulted and which terms were assumed is the
        part of a trace a disputed number is actually read for."""
        store.record(
            _record(
                "abc123",
                trios=["trio-loyalty"],
                assumptions=["loyal", "top"],
                preference_changes=[("answer_format", "bullets")],
            )
        )

        found = store.get(owner_id="dana", turn_id="abc123")

        assert found.trios == ["trio-loyalty"]
        assert found.assumptions == ["loyal", "top"]
        assert found.preference_changes == [("answer_format", "bullets")]

    def test_a_trace_with_no_reasons_reads_back_empty_rather_than_null(self, store):
        """Rows written before these columns existed, and turns that consulted
        nothing, must not come back as None and break the renderer."""
        store.record(_record("abc123"))

        found = store.get(owner_id="dana", turn_id="abc123")

        assert found.trios == []
        assert found.assumptions == []
        assert found.preference_changes == []

    def test_a_trace_is_scoped_to_its_owner(self, store):
        """Turn ids are short. Another user's trace holds their question text."""
        store.record(_record("abc123"))

        assert store.get(owner_id="sam", turn_id="abc123") is None

    def test_unknown_turn_id_returns_none(self, store):
        assert store.get(owner_id="dana", turn_id="nope") is None

    def test_recent_returns_newest_first_for_that_owner_only(self, store):
        store.record(_record("t1", question="first"))
        store.record(_record("t2", question="second"))
        store.record(_record("t3", owner_id="sam", question="not mine"))

        recent = store.recent(owner_id="dana", limit=10)

        assert [t.question for t in recent] == ["second", "first"]

    def test_recording_the_same_turn_twice_does_not_duplicate_it(self, store):
        """A resumed turn is persisted once completed; a retry must not double
        it, or every metric computed per turn is wrong."""
        store.record(_record("abc123", status="ok"))
        store.record(_record("abc123", status="failed"))

        recent = store.recent(owner_id="dana", limit=10)

        assert len(recent) == 1
        assert recent[0].status == "failed", "the later record wins"

    def test_metrics_over_recorded_turns(self, store):
        store.record(_record("t1", status="ok", redactions=2, bytes_billed=1000))
        store.record(_record("t2", status="failed", redactions=0, bytes_billed=3000))

        metrics = store.metrics(owner_id="dana")

        assert metrics["turns"] == 2
        assert metrics["bytes_billed"] == 4000
        assert metrics["redactions"] == 2

    def test_metrics_on_an_empty_store_do_not_divide_by_zero(self, store):
        metrics = store.metrics(owner_id="dana")

        assert metrics["turns"] == 0
        assert metrics["first_pass_validity"] == 0.0

    def test_first_pass_validity_counts_turns_whose_first_draft_survived(self, store):
        clean = [{"step_id": "step_1", "violations": [], "error": None}]
        rejected = [
            {"step_id": "step_1", "violations": ["Column 'email' is personal data."]},
            {"step_id": "step_1", "violations": [], "error": None},
        ]
        store.record(_record("t1", attempts=clean))
        store.record(_record("t2", attempts=rejected))

        metrics = store.metrics(owner_id="dana")

        assert metrics["first_pass_validity"] == 0.5

    def test_node_latency_is_reported_per_node(self, store):
        """Three samples, so the median is an observation rather than a mean of
        two — the assertion says what the statistic is."""
        for turn, (route_ms, execute_ms) in enumerate(
            [(100, 500), (300, 900), (500, 1300)], start=1
        ):
            store.record(
                _record(
                    f"t{turn}",
                    events=[("route", route_ms, ""), ("execute", execute_ms, "")],
                )
            )

        metrics = store.metrics(owner_id="dana")

        assert metrics["node_p50_ms"]["route"] == 300
        assert metrics["node_p50_ms"]["execute"] == 900
