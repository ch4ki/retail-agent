import pytest

from retail_agent.config import Settings
from retail_agent.llm.provider import MissingCredentialsError, build_llm


def test_gemini_requires_a_key():
    settings = Settings(_env_file=None, llm_provider="gemini", google_api_key=None)
    with pytest.raises(MissingCredentialsError) as excinfo:
        build_llm(settings)
    assert "GOOGLE_API_KEY" in str(excinfo.value)


def test_openai_requires_a_key():
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key=None)
    with pytest.raises(MissingCredentialsError):
        build_llm(settings)


def test_openrouter_requires_a_key():
    settings = Settings(
        _env_file=None, llm_provider="openrouter", openrouter_api_key=None
    )
    with pytest.raises(MissingCredentialsError):
        build_llm(settings)


def test_ollama_needs_no_key():
    settings = Settings(_env_file=None, llm_provider="ollama")
    assert build_llm(settings) is not None


def test_gemini_uses_the_default_model_when_unset():
    settings = Settings(_env_file=None, llm_provider="gemini", google_api_key="fake")
    model = build_llm(settings)

    assert settings.resolved_model == "gemini-2.5-flash"
    assert model is not None


def test_explicit_model_overrides_the_default():
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key="fake",
        llm_model="gemini-2.5-pro",
    )
    assert settings.resolved_model == "gemini-2.5-pro"
    assert build_llm(settings) is not None


def test_error_message_names_the_fix():
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key=None)
    with pytest.raises(MissingCredentialsError) as excinfo:
        build_llm(settings)

    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert ".env" in message
