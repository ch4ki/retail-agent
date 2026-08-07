"""The signal store and the proposal rules.

Detection itself moved to the router — see `tests/unit/test_style_signal.py` for
the validation that keeps a proposal quotable, and `-m live` for whether the
real model actually declines to read "why are sales down?" as a style
preference.
"""

import pytest

from retail_agent.store.learning import (
    DECLINED_MULTIPLIER,
    PROPOSAL_THRESHOLD,
    InMemorySignalStore,
    Signal,
    SignalStore,
    next_proposal,
)
from retail_agent.store.preferences import Preferences
from tests.support.signal_store_contract import SignalStoreContract


# --- the store ---


def test_satisfies_the_protocol():
    assert isinstance(InMemorySignalStore(), SignalStore)


def test_evidence_accumulates_per_user():
    store = InMemorySignalStore()
    signal = Signal(field="depth", value="summary", evidence="keep it brief")

    store.record(user_id="dana", signal=signal)
    store.record(user_id="dana", signal=signal)
    store.record(user_id="sam", signal=signal)

    assert store.counts(user_id="dana")[("depth", "summary")][0] == 2
    assert store.counts(user_id="sam")[("depth", "summary")][0] == 1


# --- proposing ---


def _record(store, times, field="depth", value="summary", evidence="keep it brief"):
    for _ in range(times):
        store.record(
            user_id="dana", signal=Signal(field=field, value=value, evidence=evidence)
        )


def test_nothing_is_proposed_below_the_threshold():
    """Two is a coincidence."""
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD - 1)

    assert next_proposal(store, user_id="dana", current=Preferences()) is None


def test_a_proposal_appears_at_the_threshold():
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD)

    proposal = next_proposal(store, user_id="dana", current=Preferences())

    assert proposal.field == "depth"
    assert proposal.value == "summary"
    assert proposal.count == PROPOSAL_THRESHOLD


def test_the_question_quotes_the_evidence_and_the_count():
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD, evidence="just give me the numbers")

    question = next_proposal(store, user_id="dana", current=Preferences()).question()

    assert "just give me the numbers" in question
    assert str(PROPOSAL_THRESHOLD) in question
    assert "/prefs accept" in question and "/prefs decline" in question


def test_nothing_is_proposed_for_a_setting_already_chosen():
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD * 2)

    current = Preferences(depth="summary")

    assert next_proposal(store, user_id="dana", current=current) is None


def test_a_declined_suggestion_needs_much_more_evidence():
    """Being asked twice about something you already refused is how a helpful
    feature becomes an irritating one."""
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD)
    store.decline(user_id="dana", field="depth", value="summary")

    assert next_proposal(store, user_id="dana", current=Preferences()) is None

    _record(store, PROPOSAL_THRESHOLD * DECLINED_MULTIPLIER)
    assert next_proposal(store, user_id="dana", current=Preferences()) is not None


def test_the_strongest_signal_is_proposed_first():
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD)
    _record(store, PROPOSAL_THRESHOLD + 4, field="answer_format", value="bullets",
            evidence="use bullets")

    proposal = next_proposal(store, user_id="dana", current=Preferences())

    assert proposal.field == "answer_format"


def test_accepting_clears_the_evidence_that_produced_it():
    """Otherwise the counters that triggered the proposal immediately trigger
    it again for the opposite value."""
    store = InMemorySignalStore()
    _record(store, PROPOSAL_THRESHOLD)

    store.clear(user_id="dana", field="depth")

    assert next_proposal(store, user_id="dana", current=Preferences()) is None


# --- the same contract, in memory ---


class TestInMemorySignalStore(SignalStoreContract):
    @pytest.fixture
    def store(self):
        return InMemorySignalStore()
