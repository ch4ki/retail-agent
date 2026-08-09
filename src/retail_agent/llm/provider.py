"""Provider factory. One env var switches the whole agent between backends.

This builds models and nothing else. Retry and fallback are middleware now
(`agent/middleware.py`), so what a caller gets back here is an ordinary
`BaseChatModel` — the primary — plus the fallbacks the middleware needs.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from retail_agent.config import Settings

log = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    """Provider selected but its key is absent."""


def build_llm(settings: Settings) -> BaseChatModel:
    """The primary provider — the model every call starts at.

    Retry no longer lives in the object this returns, so a caller that only
    takes this is calling one provider once. That is fine for the agent, whose
    middleware supplies both, and it is the accepted cost for the one direct
    model call outside a graph (`knowledge/proposals.py`), which already treats
    any failure as "no options to offer".
    """
    return build_chain(settings)[0][1]


def build_models(settings: Settings) -> tuple[BaseChatModel, list[BaseChatModel]]:
    """The primary and its fallbacks, from one pass over the chain.

    Both come back together because building them separately would construct
    the primary twice — two client objects, and the missing-credential warnings
    for every fallback logged twice with it.
    """
    chain = build_chain(settings)
    return chain[0][1], [model for _, model in chain[1:]]


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
