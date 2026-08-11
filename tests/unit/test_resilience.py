"""Which provider failures are worth waiting for, and the two ways of acting
on it.

The brief asks for resilience to third-party downtime, and a single-vendor
agent cannot answer that honestly. Inside an agent loop, that is langchain's
`ModelRetryMiddleware` and `ModelFallbackMiddleware`; outside one, it is
`resilient_call`, a hand-written copy of the same policy for `report_writer`,
`ask_about_report` and `propose`, which are single model calls with no tool
loop for middleware to wrap. This file tests the judgement both paths share —
`is_retryable` — the wiring that hands it to the middleware pair, and
`resilient_call` itself. The middleware pair's own behaviour is covered end to
end in `tests/component/test_supervisor.py`, through a real compiled agent.
"""

import pytest
from types import SimpleNamespace

from retail_agent.config import Settings
from retail_agent.llm.resilience import MAX_DELAY_SECONDS, is_retryable, resilient_call


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


# --- calls that happen outside an agent loop ---
#
# `report_writer`, `ask_about_report` and `propose` are single model calls with
# no tool loop, so `create_agent`'s middleware cannot reach them. These pin that
# `resilient_call` reproduces the middleware pair's semantics exactly — the two
# implementations are kept honest by these assertions and nothing else.


class _Provider:
    """A model that fails a scripted number of times, then answers."""

    def __init__(self, name, *, fails_with=()):
        self.name = name
        self.fails_with = list(fails_with)
        self.attempts = 0

    def respond(self):
        self.attempts += 1
        if self.fails_with:
            raise self.fails_with.pop(0)
        return f"{self.name}-ok"


def _call(model):
    return model.respond()


def _deps_for(primary, *fallbacks, attempts=3):
    return SimpleNamespace(
        llm=primary,
        llm_fallbacks=list(fallbacks),
        settings=Settings(_env_file=None, llm_retry_attempts=attempts),
    )


def test_a_transient_failure_is_retried_and_then_succeeds():
    provider = _Provider("primary", fails_with=[RuntimeError("429 rate limit")])
    slept = []

    assert resilient_call(_deps_for(provider), _call, sleep=slept.append) == "primary-ok"
    assert provider.attempts == 2
    assert slept == [0.5]


def test_a_permanent_failure_costs_one_attempt_then_the_next_provider():
    """The behaviour `retry_on` exists for: a rejected key must not spend the
    whole budget before reaching a provider that would have answered."""
    primary = _Provider("primary", fails_with=[RuntimeError("401 invalid api key")] * 5)
    backup = _Provider("backup")
    slept = []

    result = resilient_call(_deps_for(primary, backup), _call, sleep=slept.append)

    assert result == "backup-ok"
    assert primary.attempts == 1
    assert slept == [], "nothing was waited for"


def test_retries_are_exhausted_against_one_provider_before_the_next():
    """Fallback is outermost in the middleware stack for this reason; the loop
    order here is what reproduces it."""
    primary = _Provider("primary", fails_with=[RuntimeError("503 unavailable")] * 5)
    backup = _Provider("backup")

    result = resilient_call(_deps_for(primary, backup), _call, sleep=lambda _: None)

    assert result == "backup-ok"
    assert primary.attempts == 3


def test_every_provider_exhausted_raises_the_last_error():
    primary = _Provider("primary", fails_with=[RuntimeError("503 unavailable")] * 5)
    backup = _Provider("backup", fails_with=[RuntimeError("504 gateway timeout")] * 5)

    with pytest.raises(RuntimeError, match="504"):
        resilient_call(_deps_for(primary, backup), _call, sleep=lambda _: None)


def test_the_attempt_count_comes_from_settings():
    """`llm_retry_attempts` is total attempts, not retries-after-the-first —
    the same off-by-one `max_retries == attempts - 1` guards for the middleware."""
    primary = _Provider("primary", fails_with=[RuntimeError("429 rate limit")] * 9)
    backup = _Provider("backup")

    resilient_call(_deps_for(primary, backup, attempts=1), _call, sleep=lambda _: None)

    assert primary.attempts == 1


def test_the_backoff_grows_and_is_capped():
    primary = _Provider("primary", fails_with=[RuntimeError("429 rate limit")] * 9)
    slept = []

    with pytest.raises(RuntimeError):
        resilient_call(_deps_for(primary, attempts=8), _call, sleep=slept.append)

    assert slept == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    assert max(slept) == MAX_DELAY_SECONDS


def test_providers_are_tried_primary_first():
    primary = _Provider("primary", fails_with=[RuntimeError("503 unavailable")])
    backup = _Provider("backup")
    order = []

    def watching(model):
        order.append(model.name)
        return model.respond()

    resilient_call(
        _deps_for(primary, backup, attempts=1), watching, sleep=lambda _: None
    )

    assert order == ["primary", "backup"]
