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
