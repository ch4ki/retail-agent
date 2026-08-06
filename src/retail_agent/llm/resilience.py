"""Retry, fall back, and stop calling a provider that is down.

The brief asks for resilience to third-party downtime. A single-vendor agent
cannot answer that honestly, so provider selection is a chain rather than a
setting: transient failures are retried on the current provider, permanent ones
move straight to the next, and a provider that keeps failing is skipped
entirely until a cooldown elapses.

The whole chain sits behind the two methods the nodes actually use — `invoke`
and `with_structured_output` — so nothing in `agent/` knows this exists.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

# Worth waiting for: the provider is up but busy, slow, or briefly unreachable.
_TRANSIENT = (
    "429",
    "resource_exhausted",
    "rate limit",
    "quota",
    "timeout",
    "timed out",
    "connection",
    "unreachable",
    "temporarily",
    "overloaded",
    "unavailable",
    "500",
    "502",
    "503",
    "504",
)

# Not worth waiting for: the same call will be rejected the same way.
_PERMANENT = (
    "401",
    "403",
    "invalid api key",
    "api key not valid",
    "unauthenticated",
    "permission denied",
    "400",
    "invalid_request",
)

BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 8.0


def is_retryable(err: Exception) -> bool:
    """Whether trying the same provider again could plausibly work.

    Permanent markers are checked first: a rejected key often arrives in an
    envelope that also mentions a connection, and retrying it just adds latency
    before the identical failure.
    """
    lowered = str(err).lower()
    if any(marker in lowered for marker in _PERMANENT):
        return False
    return any(marker in lowered for marker in _TRANSIENT)


class CircuitBreaker:
    """Consecutive failures per provider, with a cooldown.

    Deliberately never refuses every provider — see `ResilientChatModel`. A
    breaker that can leave nothing available converts a degraded agent into a
    dead one, which is a worse failure than a slow call.
    """

    def __init__(
        self,
        *,
        threshold: int = 3,
        cooldown_seconds: float = 60.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._now = now
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def record_failure(self, provider: str) -> None:
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= max(1, self._threshold):
            self._opened_at[provider] = self._now()

    def record_success(self, provider: str) -> None:
        self._failures.pop(provider, None)
        self._opened_at.pop(provider, None)

    def is_open(self, provider: str) -> bool:
        opened = self._opened_at.get(provider)
        if opened is None:
            return False
        if self._now() - opened >= self._cooldown:
            # Cooldown elapsed: let one call through and judge by its outcome.
            self._opened_at.pop(provider, None)
            self._failures.pop(provider, None)
            return False
        return True


class ResilientChatModel:
    """A chat model backed by an ordered chain of providers."""

    def __init__(
        self,
        providers: Sequence[tuple[str, object]],
        *,
        attempts: int = 3,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not providers:
            raise ValueError("a chain needs at least one provider")
        self._providers = list(providers)
        self._attempts = max(1, attempts)
        self._breaker = breaker or CircuitBreaker()
        self._sleep = sleep

    @property
    def primary(self):
        return self._providers[0][1]

    def invoke(self, messages, **kwargs):
        return self._call(lambda model: model.invoke(messages, **kwargs))

    def with_structured_output(self, schema, **kwargs):
        """Routing and planning go through here, so it needs the same chain."""
        return _ResilientStructured(self, schema, kwargs)

    def _call(self, run):
        last_error: Exception | None = None

        for name, model in self._usable():
            for attempt in range(1, self._attempts + 1):
                try:
                    result = run(model)
                except Exception as err:  # provider SDKs raise their own types
                    last_error = err
                    self._breaker.record_failure(name)

                    if not is_retryable(err) or attempt == self._attempts:
                        log.warning(
                            "provider %s failed (%s); %s",
                            name,
                            type(err).__name__,
                            "trying the next provider" if self._has_next(name) else "no fallback left",
                        )
                        break

                    delay = self._backoff(attempt)
                    log.info("provider %s transient failure; retrying in %.1fs", name, delay)
                    self._sleep(delay)
                else:
                    self._breaker.record_success(name)
                    return result

        assert last_error is not None  # _usable() never yields nothing
        raise last_error

    def _usable(self):
        """Closed providers first; if the breaker has opened every one of them,
        fall back to trying them all rather than failing without a call."""
        closed = [p for p in self._providers if not self._breaker.is_open(p[0])]
        return closed or self._providers

    def _has_next(self, name: str) -> bool:
        names = [n for n, _ in self._providers]
        return names.index(name) < len(names) - 1

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential with jitter, so retries from concurrent turns spread out
        instead of arriving together and re-triggering the same rate limit."""
        ceiling = min(BASE_DELAY_SECONDS * (2 ** (attempt - 1)), MAX_DELAY_SECONDS)
        return ceiling * (0.5 + random.random() / 2)


class _ResilientStructured:
    def __init__(self, resilient: ResilientChatModel, schema, kwargs: dict) -> None:
        self._resilient = resilient
        self._schema = schema
        self._kwargs = kwargs

    def invoke(self, messages, **kwargs):
        return self._resilient._call(
            lambda model: model.with_structured_output(
                self._schema, **self._kwargs
            ).invoke(messages, **kwargs)
        )
