"""Provider factory. One env var switches the whole agent between backends.

Retry, fallback chains and the circuit breaker arrive in phase 2; this file is
the seam they attach to.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from retail_agent.config import Settings


class MissingCredentialsError(RuntimeError):
    """Provider selected but its key is absent."""


def build_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    model = settings.resolved_model
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
