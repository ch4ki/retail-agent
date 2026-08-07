"""One contract, two implementations.

The properties that matter here are about *accumulation across sessions*. The
proposal threshold is three, so evidence that dies with the process means the
agent can only ever notice a preference expressed three times in one sitting —
which is the bug that made this feature nominal rather than real.
"""

import pytest

from retail_agent.store.learning import Signal


def signal(field="depth", value="summary", evidence="cut to the chase"):
    return Signal(field=field, value=value, evidence=evidence)


class SignalStoreContract:
    @pytest.fixture
    def store(self):
        raise NotImplementedError

    def test_the_first_sighting_counts_as_one(self, store):
        assert store.record(user_id="dana", signal=signal()) == 1

    def test_evidence_accumulates(self, store):
        store.record(user_id="dana", signal=signal())
        store.record(user_id="dana", signal=signal())

        assert store.record(user_id="dana", signal=signal()) == 3

    def test_the_latest_wording_is_the_one_quoted(self, store):
        """The proposal says "most recently ...", so the newest phrasing wins —
        quoting something from three sessions ago reads as stale."""
        store.record(user_id="dana", signal=signal(evidence="keep it brief"))
        store.record(user_id="dana", signal=signal(evidence="cut to the chase"))

        assert store.counts(user_id="dana")[("depth", "summary")] == (2, "cut to the chase")

    def test_users_do_not_share_evidence(self, store):
        """Two managers using one deployment must not train each other's
        preferences."""
        store.record(user_id="dana", signal=signal())
        store.record(user_id="sam", signal=signal())
        store.record(user_id="sam", signal=signal())

        assert store.counts(user_id="dana")[("depth", "summary")][0] == 1
        assert store.counts(user_id="sam")[("depth", "summary")][0] == 2

    def test_fields_are_counted_separately(self, store):
        store.record(user_id="dana", signal=signal())
        store.record(user_id="dana", signal=signal(field="answer_format", value="table"))

        assert set(store.counts(user_id="dana")) == {
            ("depth", "summary"),
            ("answer_format", "table"),
        }

    def test_an_unknown_user_has_no_evidence(self, store):
        assert store.counts(user_id="nobody") == {}

    def test_a_decline_is_remembered(self, store):
        store.decline(user_id="dana", field="depth", value="summary")

        assert store.declines(user_id="dana") == {("depth", "summary"): 1}

    def test_declines_accumulate(self, store):
        """Each refusal multiplies the evidence needed before asking again."""
        store.decline(user_id="dana", field="depth", value="summary")
        store.decline(user_id="dana", field="depth", value="summary")

        assert store.declines(user_id="dana")[("depth", "summary")] == 2

    def test_declines_are_per_user(self, store):
        store.decline(user_id="dana", field="depth", value="summary")

        assert store.declines(user_id="sam") == {}

    def test_clearing_a_field_drops_its_evidence(self, store):
        """Once a setting is decided, the evidence that produced it stops
        mattering — otherwise accepting leaves the counters that triggered it."""
        store.record(user_id="dana", signal=signal())
        store.record(user_id="dana", signal=signal(field="answer_format", value="table"))

        store.clear(user_id="dana", field="depth")

        assert set(store.counts(user_id="dana")) == {("answer_format", "table")}

    def test_clearing_leaves_other_users_alone(self, store):
        store.record(user_id="dana", signal=signal())
        store.record(user_id="sam", signal=signal())

        store.clear(user_id="dana", field="depth")

        assert store.counts(user_id="sam")

    def test_clearing_does_not_erase_a_decline(self, store):
        """Accepting a different setting must not forget that this one was
        refused, or the next proposal arrives at full strength."""
        store.decline(user_id="dana", field="depth", value="summary")

        store.clear(user_id="dana", field="depth")

        assert store.declines(user_id="dana")[("depth", "summary")] == 1
