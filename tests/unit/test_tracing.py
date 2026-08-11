import os

import pytest

from retail_agent.config import Settings
from retail_agent.obs.tracing import configure_tracing


@pytest.fixture(autouse=True)
def _clean_langsmith_env(monkeypatch):
    for var in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(var, raising=False)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_disabled_by_default():
    assert configure_tracing(settings()) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_enabling_without_a_key_stays_off():
    # Turning tracing on with no key makes every LLM call warn. Refuse quietly.
    assert configure_tracing(settings(langsmith_tracing=True)) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_key_without_the_flag_stays_off():
    assert configure_tracing(settings(langsmith_api_key="ls-key")) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_enabled_when_flag_and_key_are_both_present():
    enabled = configure_tracing(
        settings(langsmith_tracing=True, langsmith_api_key="ls-key")
    )

    assert enabled is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_PROJECT"] == "retail-agent"


def test_project_name_is_configurable():
    configure_tracing(
        settings(
            langsmith_tracing=True,
            langsmith_api_key="ls-key",
            langsmith_project="retail-agent-dev",
        )
    )

    assert os.environ["LANGSMITH_PROJECT"] == "retail-agent-dev"


def test_endpoint_is_only_set_when_given():
    configure_tracing(settings(langsmith_tracing=True, langsmith_api_key="k"))
    assert "LANGSMITH_ENDPOINT" not in os.environ

    configure_tracing(
        settings(
            langsmith_tracing=True,
            langsmith_api_key="k",
            langsmith_endpoint="https://eu.api.smith.langchain.com",
        )
    )
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


def test_langsmith_agrees_that_tracing_is_on():
    from langsmith import utils

    configure_tracing(settings(langsmith_tracing=True, langsmith_api_key="ls-key"))

    assert utils.tracing_is_enabled() is True


def test_context_metrics_ignore_turns_that_were_never_measured():
    """A turn that died never reached the recorder and carries 0. Counting
    those would drag the median below the number a threshold is set against."""
    from retail_agent.obs.traces import TraceRecord, compute_metrics

    def trace(turn_id, tokens, status="ok"):
        return TraceRecord(
            turn_id=turn_id,
            session_id="s1",
            owner_id="exec",
            question="q",
            intent="analyze",
            status=status,
            context_tokens=tokens,
        )

    metrics = compute_metrics(
        [trace("a", 1_000), trace("b", 3_000), trace("c", 0, "failed")]
    )

    assert metrics["context_tokens_max"] == 3_000
    assert metrics["context_tokens_p50"] == 2_000


def test_duration_ms_sums_every_events_timing():
    """Previously untested: `duration_ms=sum(event["ms"] for event in events)`
    in `trace_from_state` would pass every other test unchanged if it summed
    nothing, or only the first event, instead of all of them."""
    from retail_agent.obs.traces import trace_from_state

    state = {
        "messages": [],
        "events": [
            {"name": "lookup_definitions", "ms": 40, "detail": ""},
            {"name": "run_sql", "ms": 120, "detail": ""},
            {"name": "run_sql", "ms": 80, "detail": ""},
        ],
        "attempts": [],
    }

    trace = trace_from_state(
        state, "answer", user_id="exec", session_id="s1", turn_id="t1"
    )

    assert trace.duration_ms == 240


def test_bytes_billed_sums_every_attempts_billing():
    """Previously untested: `bytes_billed=sum(a.get("bytes_billed") or 0 for a
    in attempts)` would pass every other test unchanged if it summed nothing,
    or only the first attempt's, instead of every query a turn ran."""
    from retail_agent.obs.traces import trace_from_state

    state = {
        "messages": [],
        "events": [],
        "attempts": [
            {"step_id": "q1", "sql": "SELECT 1", "bytes_billed": 1_000},
            {"step_id": "q2", "sql": "SELECT 2", "bytes_billed": 500},
        ],
    }

    trace = trace_from_state(
        state, "answer", user_id="exec", session_id="s1", turn_id="t1"
    )

    assert trace.bytes_billed == 1_500


def test_context_metrics_are_zero_with_nothing_measured():
    """`/metrics` must not divide by zero on a session of failed turns."""
    from retail_agent.obs.traces import TraceRecord, compute_metrics

    metrics = compute_metrics(
        [
            TraceRecord(
                turn_id="a",
                session_id="s1",
                owner_id="exec",
                question="q",
                intent="chat",
                status="failed",
            )
        ]
    )

    assert metrics["context_tokens_max"] == 0
    assert metrics["context_tokens_p50"] == 0
