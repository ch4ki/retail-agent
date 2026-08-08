"""Provider factory. One env var switches the whole agent between backends.

`build_llm` returns the primary alone when nothing else is configured, and a
`ResilientChatModel` over the whole chain when it is — so the retry, fallback
and circuit-breaker behaviour is opt-in per deployment and invisible to every
caller either way.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from retail_agent.config import Settings

log = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    """Provider selected but its key is absent."""


def build_llm(settings: Settings) -> BaseChatModel:
    """The model the agent calls: one provider, or a chain over several.

    Always wrapped, including for a single provider. This used to return the
    bare model whenever the chain had one entry — the reasoning being that with
    nothing to fall back to there was nothing to do. But retry and the circuit
    breaker are not about fallback, so `llm_retry_attempts` was configured and
    silently ignored for the most common deployment there is.

    Measured: a transient `Connection error.` ended the turn rather than being
    retried, and a single blip cost 37 of 47 cases in a live eval run.
    `_TRANSIENT` has always listed "connection". Nothing was asking it.
    """
    chain = build_chain(settings)

    from retail_agent.llm.resilience import CircuitBreaker, ResilientChatModel

    return ResilientChatModel(
        chain,
        attempts=settings.llm_retry_attempts,
        breaker=CircuitBreaker(
            threshold=settings.llm_breaker_threshold,
            cooldown_seconds=settings.llm_breaker_cooldown_seconds,
        ),
    )


def build_chain(settings: Settings) -> list[tuple[str, BaseChatModel]]:
    """The primary first, then each usable fallback.

    A fallback whose credentials are missing is dropped with a warning: naming
    a provider you have no key for should cost you that fallback, not the
    agent. The primary is never dropped — a missing primary key is a
    configuration error the user needs to see at startup.
    """
    chain: list[tuple[str, BaseChatModel]] = [
        (settings.llm_provider, build_one(settings, settings.llm_provider))
    ]
    seen = {settings.llm_provider}

    for name in (p.strip() for p in settings.llm_fallbacks.split(",")):
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            chain.append((name, build_one(settings, name)))
        except (MissingCredentialsError, ValueError) as err:
            log.warning("fallback provider %s unavailable: %s", name, err)

    return chain


def build_one(settings: Settings, provider: str) -> BaseChatModel:
    model = settings.model_for(provider)
    temperature = settings.llm_temperature
    max_tokens = settings.llm_max_tokens

    if provider == "gemini":
        _require(settings.google_api_key, "GOOGLE_API_KEY", provider)
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "openai":
        _require(settings.openai_api_key, "OPENAI_API_KEY", provider)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openrouter":
        _require(settings.openrouter_api_key, "OPENROUTER_API_KEY", provider)
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _require(value: str | None, env_var: str, provider: str) -> None:
    if not value:
        raise MissingCredentialsError(
            f"LLM_PROVIDER={provider} needs {env_var}. "
            f"Add it to .env, or pick a different provider."
        )
