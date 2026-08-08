"""Provider failure handling.

The brief asks for resilience to third-party downtime, and a single-vendor
agent cannot answer that honestly. These use fake models, so no network and no
sleeping: the backoff is injected.
"""

import pytest

from retail_agent.llm.resilience import (
    CircuitBreaker,
    ResilientChatModel,
    is_retryable,
)


class FlakyModel:
    """Raises the queued errors in order, then answers."""

    def __init__(self, name, errors=(), answer="ok"):
        self.name = name
        self.errors = list(errors)
        self.answer = answer
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return f"{self.name}:{self.answer}"

    def with_structured_output(self, schema, **kwargs):
        return _Structured(self, schema)


class _Structured:
    def __init__(self, model, schema):
        self.model = model
        self.schema = schema

    def invoke(self, messages, **kwargs):
        self.model.invoke(messages)
        return self.schema


def build(*models, **kwargs):
    kwargs.setdefault("sleep", lambda _seconds: None)
    return ResilientChatModel(
        [(m.name, m) for m in models], attempts=3, **kwargs
    )


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


# --- retry ---


def test_a_transient_failure_is_retried_on_the_same_provider():
    model = FlakyModel("gemini", errors=[RuntimeError("429 rate limited")])

    assert build(model).invoke("hi") == "gemini:ok"
    assert model.calls == 2


def test_retries_are_bounded_then_fall_through():
    primary = FlakyModel("gemini", errors=[RuntimeError("429")] * 5)
    backup = FlakyModel("openai")

    assert build(primary, backup).invoke("hi") == "openai:ok"
    assert primary.calls == 3, "attempts=3, then move on"


def test_backoff_grows_and_is_jittered():
    slept = []
    primary = FlakyModel("gemini", errors=[RuntimeError("429")] * 5)
    backup = FlakyModel("openai")

    build(primary, backup, sleep=slept.append).invoke("hi")

    assert len(slept) == 2, "one wait between each pair of attempts"
    assert slept[1] > slept[0], "exponential"
    assert all(0 < s < 10 for s in slept), "bounded"


# --- fallback ---


def test_a_permanent_failure_skips_straight_to_the_next_provider():
    primary = FlakyModel("gemini", errors=[RuntimeError("401 invalid api key")])
    backup = FlakyModel("openai")

    assert build(primary, backup).invoke("hi") == "openai:ok"
    assert primary.calls == 1, "no point retrying a rejected key"


def test_the_last_provider_failing_raises_the_last_error():
    """Degrading to silence is worse than an error the CLI can explain."""
    primary = FlakyModel("gemini", errors=[RuntimeError("401 bad key")])
    backup = FlakyModel("ollama", errors=[RuntimeError("connection refused")] * 5)

    with pytest.raises(Exception) as excinfo:
        build(primary, backup).invoke("hi")

    assert "refused" in str(excinfo.value)


def test_structured_output_falls_back_too():
    """Routing and planning go through with_structured_output, so a fallback
    that only covered invoke would leave both unprotected."""

    class Schema:
        pass

    primary = FlakyModel("gemini", errors=[RuntimeError("401")])
    backup = FlakyModel("openai")

    result = build(primary, backup).with_structured_output(Schema).invoke("hi")

    assert result is Schema
    assert backup.calls == 1


# --- circuit breaker ---


class Clock:
    """A clock the test moves deliberately.

    An iterator of timestamps couples the assertions to how many times the
    implementation happens to read the clock, which is not behaviour worth
    pinning.
    """

    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds


def test_the_breaker_opens_after_repeated_failures():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60, now=Clock())

    breaker.record_failure("gemini")
    assert breaker.is_open("gemini") is False, "one failure is not an outage"
    breaker.record_failure("gemini")
    assert breaker.is_open("gemini") is True


def test_the_breaker_closes_after_the_cooldown():
    clock = Clock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=60, now=clock)

    breaker.record_failure("gemini")
    assert breaker.is_open("gemini") is True

    clock.seconds = 61
    assert breaker.is_open("gemini") is False, "cooldown elapsed"


def test_success_closes_the_breaker():
    breaker = CircuitBreaker(threshold=2, cooldown_seconds=60, now=lambda: 0)

    breaker.record_failure("gemini")
    breaker.record_success("gemini")
    breaker.record_failure("gemini")

    assert breaker.is_open("gemini") is False, "the streak was broken"


def test_an_open_provider_is_skipped_without_being_called():
    """The point of the breaker: stop paying latency for a provider that is
    known to be down."""
    primary = FlakyModel("gemini", errors=[RuntimeError("503")] * 20)
    backup = FlakyModel("openai")
    resilient = build(primary, backup, breaker=CircuitBreaker(threshold=1, now=lambda: 0))

    resilient.invoke("first")
    calls_after_first = primary.calls
    resilient.invoke("second")

    assert primary.calls == calls_after_first, "not called again while open"
    assert backup.calls == 2


def test_every_provider_open_still_attempts_rather_than_giving_up():
    """A breaker that can refuse every provider turns a slow agent into a dead
    one. When nothing is available, try anyway."""
    only = FlakyModel("gemini")
    resilient = build(only, breaker=CircuitBreaker(threshold=0, now=lambda: 0))

    assert resilient.invoke("hi") == "gemini:ok"


# --- tool binding, which is how `create_agent` compiles ---------------------


class ToolBindable(FlakyModel):
    """A provider that records what it was bound to.

    `FlakyModel` deliberately has no `bind_tools`, because most of this file is
    about the calling behaviour. This subclass exists for the compile path.
    """

    def __init__(self, name, errors=(), answer="ok"):
        super().__init__(name, errors=errors, answer=answer)
        self.bound = None

    def bind_tools(self, tools, **kwargs):
        bound = ToolBindable(f"{self.name}+tools", errors=self.errors, answer=self.answer)
        bound.bound = list(tools)
        self.bound = list(tools)
        return bound


def test_binding_tools_keeps_the_whole_chain():
    """The bug this exists for: `create_agent` compiles by calling `bind_tools`,
    and without it the agent raises `AttributeError` at the first model call.

    Every offline test passed while that was true, because the doubles are bound
    directly rather than through the chain. It took a live run to find.
    """
    primary, fallback = ToolBindable("a"), ToolBindable("b")
    chain = ResilientChatModel([("a", primary), ("b", fallback)], sleep=lambda _: None)

    bound = chain.bind_tools([print])

    assert isinstance(bound, ResilientChatModel)
    assert primary.bound == [print]
    assert fallback.bound == [print], (
        "a fallback that lost its tools is worse than the outage it recovers from"
    )


def test_a_bound_chain_still_falls_back():
    primary = ToolBindable("a", errors=[RuntimeError("503 unavailable")] * 3)
    fallback = ToolBindable("b")
    chain = ResilientChatModel([("a", primary), ("b", fallback)], sleep=lambda _: None)

    assert chain.bind_tools([print]).invoke("hi") == "b+tools:ok"


def test_the_breaker_survives_binding():
    """A fresh breaker per binding would forget an outage on every turn, and the
    agent rebuilds its model binding once per turn."""
    breaker = CircuitBreaker(threshold=1)
    breaker.record_failure("a")
    chain = ResilientChatModel(
        [("a", ToolBindable("a")), ("b", ToolBindable("b"))],
        breaker=breaker,
        sleep=lambda _: None,
    )

    assert chain.bind_tools([print]).invoke("hi") == "b+tools:ok", "skipped the open provider"


def test_a_descriptive_attribute_comes_from_the_primary():
    """Langchain reads `profile` and `_llm_type` off a model. Those describe a
    provider, not a chain, so the primary is the right answer."""
    primary = ToolBindable("a")
    primary.profile = {"max_tokens": 1}
    chain = ResilientChatModel([("a", primary), ("b", ToolBindable("b"))])

    assert chain.profile == {"max_tokens": 1}


def test_a_private_attribute_is_not_invented():
    """`__getattr__` delegating everything would make `hasattr(chain, '__deepcopy__')`
    true and break copying in ways that are very hard to trace back to here."""
    chain = ResilientChatModel([("a", ToolBindable("a"))])

    with pytest.raises(AttributeError):
        chain._not_a_real_attribute
