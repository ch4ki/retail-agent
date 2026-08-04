import pytest

from retail_agent.config import Settings


def test_defaults_do_not_require_any_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "gemini"
    assert settings.bq_dataset == "bigquery-public-data.thelook_ecommerce"
    assert settings.default_row_limit == 500
    assert "users" in settings.allowed_tables


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_ROW_LIMIT", "25")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.default_row_limit == 25


def test_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "definitely-not-a-provider")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_resolved_model_falls_back_to_provider_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert Settings(_env_file=None).resolved_model == "gemini-2.5-flash"
