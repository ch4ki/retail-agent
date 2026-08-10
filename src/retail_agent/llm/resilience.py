"""Which provider failures are worth waiting for.

The brief asks for resilience to third-party downtime, and a single-vendor
agent cannot answer that honestly. The mechanism now lives in langchain's own
middleware — `ModelRetryMiddleware` retries the current provider,
`ModelFallbackMiddleware` moves to the next one — assembled in
`agent/middleware.py`. What is left here is the one judgement those middlewares
cannot make for us: whether a given error is worth a second attempt.

This file used to hold a `ResilientChatModel` that wrapped the whole chain
behind a hand-written model interface. It was deleted because that interface
was never finished: `bind_tools`, then `bind`, then `ainvoke` each surfaced as
an `AttributeError` in front of a user, because every offline test drove the
sync path and langchain kept reaching for a method the wrapper had not thought
to implement. Middleware is handed the model rather than pretending to be one,
so that failure mode is gone rather than fixed.
"""

from __future__ import annotations

import time

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

    Passed to `ModelRetryMiddleware` as its `retry_on` predicate. The default
    there is to retry every exception, which would spend the whole budget on a
    bad API key before falling over to a provider that might have worked.
    """
    lowered = str(err).lower()
    if any(marker in lowered for marker in _PERMANENT):
        return False
    return any(marker in lowered for marker in _TRANSIENT)


def resilient_call(deps, call, *, sleep=time.sleep):
    """One model call with the resilience `create_agent` gets from middleware.

    Providers in order; retries are exhausted against one before moving to the
    next — the ordering `agent/middleware.py:_resilience` documents, reproduced
    here because middleware only exists inside an agent loop. `report_writer`,
    `ask_about_report` and `propose` are single model calls with no tool loop,
    so there is no loop for middleware to wrap.

    `call` receives a model and returns whatever it returns, rather than this
    taking messages, because one caller needs `.invoke(messages)` and another
    needs `.with_structured_output(...).invoke(prompt)`. One function, one copy
    of the policy.

    Takes `deps` rather than a model on purpose. `deps.llm` is what the agents
    hand to `ModelRetryMiddleware`; composing retry onto it here as well would
    nest retry inside retry, and the outer layer would count one logical call
    while the inner spent several.

    `sleep` is injected so tests assert the backoff without waiting for it.
    """
    models = [deps.llm, *deps.llm_fallbacks]
    attempts = max(1, deps.settings.llm_retry_attempts)
    last: Exception | None = None

    for model in models:
        for attempt in range(attempts):
            try:
                return call(model)
            except Exception as err:
                last = err
                # A permanent failure ends this provider immediately rather
                # than after the budget: the fallback below is what might
                # actually answer, and waiting first only adds latency.
                if not is_retryable(err) or attempt + 1 >= attempts:
                    break
                sleep(min(BASE_DELAY_SECONDS * 2**attempt, MAX_DELAY_SECONDS))

    raise last
