"""Turning provider exceptions into one actionable line.

Provider SDKs raise errors carrying the full JSON error envelope. Printing that
into a chat session is not graceful degradation, and it can echo back the API
key that was rejected.
"""

from __future__ import annotations

import re

KEY_ENV_VARS = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "(no key needed)",
}

# Long unbroken tokens are almost always credentials.
_SECRET = re.compile(r"\b[A-Za-z0-9_\-]{20,}\b")

MAX_LENGTH = 300


def describe_llm_error(err: Exception, *, provider: str) -> str:
    """One or two sentences the user can act on."""
    raw = str(err)
    lowered = raw.lower()

    if "resource_exhausted" in lowered or "quota" in lowered or "429" in raw:
        alternatives = " or ".join(p for p in KEY_ENV_VARS if p != provider)
        return (
            f"The {provider} API quota is exhausted. Free tiers cap daily "
            f"requests, so this usually clears on its own.\n"
            f"Wait and retry, or set LLM_PROVIDER to {alternatives} in .env."
        )

    if any(
        marker in lowered
        for marker in ("unauthenticated", "401", "api key not valid", "invalid api key")
    ):
        return (
            f"The {provider} API key was rejected. "
            f"Check {KEY_ENV_VARS.get(provider, 'the API key')} in .env."
        )

    if any(
        marker in lowered
        for marker in ("connection", "timed out", "timeout", "unreachable", "refused")
    ):
        return (
            f"Could not reach the {provider} API. Check your connection, "
            f"then retry."
        )

    return f"The {provider} API call failed: {_redact(raw)}"


def _redact(text: str) -> str:
    cleaned = _SECRET.sub("[redacted]", text)
    collapsed = " ".join(cleaned.split())
    if len(collapsed) > MAX_LENGTH:
        collapsed = f"{collapsed[:MAX_LENGTH]}…"
    return collapsed
