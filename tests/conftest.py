import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Keep tests independent of the developer's real .env and shell exports."""
    for var in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "DATABASE_URL",
        "PII_SALT",
        "DEFAULT_ROW_LIMIT",
    ):
        monkeypatch.delenv(var, raising=False)
