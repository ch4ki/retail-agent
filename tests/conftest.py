import os

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


# Written by `configure_tracing`, which is production code under test rather
# than something a test sets — so `monkeypatch` cannot be what cleans them up.
TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_TRACING_V2",
)


@pytest.fixture(autouse=True)
def _no_tracing_credentials_escape():
    """Save and restore rather than `monkeypatch.delenv`.

    `delenv` records an undo only for a name that was already present:

        if name not in dic:
            if raising: raise KeyError(name)
        else:                                   # <- only here is an undo kept
            self._setitem.append(...)

    On a machine with none of these exported that records nothing, and
    `configure_tracing` then assigns `os.environ` directly inside the test. The
    fake key survived to the end of the session, so every later test that
    invoked a model tried to POST to LangSmith and was refused — thousands of
    lines of 403s over the test output, appearing and vanishing between runs as
    `pytest-randomly` moved `test_tracing.py` around.

    Explicit here, so no test can leak these whatever it does to the
    environment.
    """
    saved = {var: os.environ.pop(var, None) for var in TRACING_VARS}
    _forget_cached_tracing_env()
    yield
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    # Restoring os.environ is not enough on its own. `langsmith.utils.get_env_var`
    # is lru_cached, so once `configure_tracing` has enabled tracing the library
    # goes on believing it whatever the environment then says — which is the very
    # thing `_invalidate_langsmith_env_cache` exists for in production.
    _forget_cached_tracing_env()


def _forget_cached_tracing_env() -> None:
    from retail_agent.obs.tracing import _invalidate_langsmith_env_cache

    _invalidate_langsmith_env_cache()
