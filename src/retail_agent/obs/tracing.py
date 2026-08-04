"""LangSmith wiring.

pydantic-settings reads `.env` into a Settings object; it does not export
anything into the process environment. LangChain's tracer only reads
os.environ, so LANGSMITH_* in `.env` has no effect until it is published here.

Call this once at startup, before any model is built.
"""

from __future__ import annotations

import logging
import os

from retail_agent.config import Settings

log = logging.getLogger(__name__)


def configure_tracing(settings: Settings) -> bool:
    """Publish LangSmith settings to the environment. Returns whether it is on.

    Tracing needs both the flag and a key. Enabling it without a key makes every
    model call emit a warning, so a half-configured setup stays off.
    """
    if not settings.langsmith_tracing:
        return False

    if not settings.langsmith_api_key:
        log.warning("LANGSMITH_TRACING is set but LANGSMITH_API_KEY is missing")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    _invalidate_langsmith_env_cache()

    log.info("LangSmith tracing enabled, project=%s", settings.langsmith_project)
    return True


def _invalidate_langsmith_env_cache() -> None:
    """Drop langsmith's cached environment reads.

    `langsmith.utils.get_env_var` is lru_cached. Importing langchain warms it
    before we get here, so the cached miss would win and nothing would be
    traced — while every other signal said tracing was on.
    """
    try:
        from langsmith import utils as ls_utils
    except ImportError:  # langsmith is optional at runtime
        return

    for name in ("get_env_var", "get_tracer_project", "get_host_url"):
        cached = getattr(ls_utils, name, None)
        clear = getattr(cached, "cache_clear", None)
        if clear is not None:
            clear()
