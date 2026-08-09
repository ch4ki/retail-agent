"""Which provider failures are worth waiting for, and how that is wired in.

The brief asks for resilience to third-party downtime, and a single-vendor
agent cannot answer that honestly. The mechanism is langchain's
`ModelRetryMiddleware` and `ModelFallbackMiddleware` now, so what is left to
test here is the judgement we still own — `is_retryable` — and the wiring that
hands it to them. The behaviour itself is covered end to end in
`tests/component/test_supervisor.py`, through a real compiled agent.
"""

import pytest

from retail_agent.llm.resilience import is_retryable


# --- classification ---


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED",
        "Connection timed out",
        "503 Service Unavailable",
        "temporarily overloaded",
    ],
)
def test_transient_failures_are_retryable(message):
    assert is_retryable(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    ["401 invalid api key", "400 bad request: schema invalid"],
)
def test_permanent_failures_are_not_retryable(message):
    """Retrying a rejected key just burns latency before the same answer."""
    assert not is_retryable(RuntimeError(message))


def test_a_permanent_marker_beats_a_transient_one_in_the_same_message():
    """A rejected key often arrives in an envelope that also says
    "connection". Retrying that spends the budget before the identical
    failure, and never reaches the provider that would have answered."""
    assert not is_retryable(RuntimeError("Connection error: 401 invalid api key"))


# --- wiring ---
#
# These assert the three settings that decide whether the pair composes at all.
# Each was wrong in an early draft, and none of them fails loudly: the agent
# still runs, it just stops falling back.


def _stack(**deps_kwargs):
    from retail_agent.agent.middleware import _resilience

    return _resilience(_FakeDeps(**deps_kwargs))


class _FakeDeps:
    def __init__(self, *, fallbacks=(), attempts=3):
        from retail_agent.config import Settings

        self.llm_fallbacks = list(fallbacks)
        self.settings = Settings(_env_file=None, llm_retry_attempts=attempts)


def test_fallback_wraps_retry_rather_than_the_other_way_round():
    """Composed handlers run first-is-outermost. Fallback has to be outermost
    so the retry budget is spent on one provider before moving to the next;
    reversed, the first retry would restart the whole sweep."""
    from langchain.agents.middleware import (
        ModelFallbackMiddleware,
        ModelRetryMiddleware,
    )

    stack = _stack(fallbacks=[object()])

    assert isinstance(stack[0], ModelFallbackMiddleware)
    assert isinstance(stack[1], ModelRetryMiddleware)


def test_retries_are_exhausted_before_the_next_provider_is_tried():
    """`max_retries` counts attempts after the first, so three total attempts
    is two retries. Off by one, the last configured attempt never happens."""
    assert _stack()[0].max_retries == 2
    assert _stack(attempts=1)[0].max_retries == 0


def test_an_exhausted_retry_raises_rather_than_answering():
    """The default is "continue", which returns an AIMessage describing the
    failure. That reads as an answer, and the fallback middleware outside it
    would never see a failure to fall back from."""
    assert _stack()[0].on_failure == "error"


def test_only_transient_failures_are_retried():
    """The default retries every exception, which spends the whole budget on a
    rejected API key."""
    assert _stack()[0].retry_on is is_retryable


def test_a_single_provider_gets_retries_but_no_fallback_layer():
    """The common deployment. Nothing should wrap the model for a fallback
    that does not exist — but `llm_retry_attempts` still has to be honoured,
    which is exactly what the old chain-of-one existed to guarantee."""
    from langchain.agents.middleware import ModelRetryMiddleware

    stack = _stack(fallbacks=[])

    assert len(stack) == 1
    assert isinstance(stack[0], ModelRetryMiddleware)
