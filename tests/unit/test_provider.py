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


def only_model(settings):
    """The single constructed provider, unwrapped.

    `build_llm` wraps every chain in `ResilientChatModel` now, including a chain
    of one. These tests are about how a provider is constructed, not about what
    wraps it, so they ask the layer that constructs it.
    """
    from retail_agent.llm.provider import build_chain

    chain = build_chain(settings)
    assert len(chain) == 1
    return chain[0][1]


# Without an explicit cap, providers reserve credit for the model's full output
# ceiling — OpenRouter rejects with a 402 before generating a token. An output
# cap is also the other half of bounding per-turn cost.


def test_openai_gets_an_output_cap():
    settings = Settings(
        _env_file=None, llm_provider="openai", openai_api_key="k", llm_max_tokens=2048
    )
    assert only_model(settings).max_tokens == 2048


def test_openrouter_gets_an_output_cap():
    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_api_key="k",
        llm_max_tokens=1024,
    )
    model = only_model(settings)
    assert model.max_tokens == 1024
    assert "openrouter.ai" in str(model.openai_api_base)


def test_gemini_gets_an_output_cap():
    settings = Settings(
        _env_file=None, llm_provider="gemini", google_api_key="k", llm_max_tokens=4096
    )
    assert only_model(settings).max_output_tokens == 4096


def test_ollama_gets_an_output_cap():
    settings = Settings(_env_file=None, llm_provider="ollama", llm_max_tokens=512)
    assert only_model(settings).num_predict == 512


def test_default_cap_is_generous_but_bounded():
    assert 1024 <= Settings(_env_file=None).llm_max_tokens <= 8192


# --- the fallback chain ---


def test_no_fallbacks_configured_still_gets_retries():
    """This asserted the opposite until a live run disproved it.

    The reasoning was that wrapping a chain of one "only adds a layer to read
    through when something breaks". But the layer is what stops it breaking:
    retry and the circuit breaker have nothing to do with fallback, so
    returning the bare model left `llm_retry_attempts` configured and ignored
    for the commonest deployment there is. One transient `Connection error.`
    then cost 37 of 47 cases in a 47-case eval.

    An extra frame in a traceback is the cheaper price.
    """
    from retail_agent.llm.provider import build_llm
    from retail_agent.llm.resilience import ResilientChatModel

    settings = Settings(_env_file=None, llm_provider="ollama", llm_fallbacks="")

    assert isinstance(build_llm(settings), ResilientChatModel)


def test_a_configured_fallback_produces_a_chain():
    from retail_agent.llm.provider import build_chain, build_llm
    from retail_agent.llm.resilience import ResilientChatModel

    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key="fake",
        llm_fallbacks="ollama",
    )

    model = build_llm(settings)

    assert isinstance(model, ResilientChatModel)
    assert [name for name, _ in build_chain(settings)] == ["gemini", "ollama"]


def test_a_fallback_without_credentials_is_dropped_not_fatal():
    """Listing a provider you have no key for should cost you that fallback,
    not the agent."""
    from retail_agent.llm.provider import build_chain

    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key="fake",
        openai_api_key=None,
        llm_fallbacks="openai,ollama",
    )

    assert [name for name, _ in build_chain(settings)] == ["gemini", "ollama"]


def test_the_primary_is_never_dropped_from_the_chain():
    """A missing primary key must still raise MissingCredentialsError at
    startup rather than silently answering from a fallback the user did not
    ask for."""
    from retail_agent.llm.provider import MissingCredentialsError, build_chain

    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key=None,
        llm_fallbacks="ollama",
    )

    with pytest.raises(MissingCredentialsError):
        build_chain(settings)


def test_the_primary_is_not_repeated_as_its_own_fallback():
    from retail_agent.llm.provider import build_chain

    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        llm_fallbacks="ollama,ollama",
    )

    assert [name for name, _ in build_chain(settings)] == ["ollama"]


def test_a_pinned_generic_model_never_reaches_a_fallback():
    """LLM_MODEL applies to the active provider only. Leaking it into the chain
    sends "gemini-2.5-pro" to OpenAI the moment Gemini goes down."""
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key="fake",
        llm_model="gemini-2.5-pro",
        llm_fallbacks="ollama",
    )

    assert settings.model_for("gemini") == "gemini-2.5-pro"
    assert settings.model_for("ollama") == "llama3.1:8b"


def test_a_per_provider_pin_is_honoured_anywhere_in_the_chain():
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        google_api_key="fake",
        ollama_model="mistral",
        llm_fallbacks="ollama",
    )

    assert settings.model_for("ollama") == "mistral"


# --- resilience must not depend on having a fallback ---


def _openrouter_only(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model")


def test_a_single_provider_still_retries(monkeypatch):
    """`build_llm` used to return the bare model whenever the chain had one
    entry, so `llm_retry_attempts` was configured and silently ignored for
    anyone running a single provider — which is most deployments.

    Measured: a transient `Connection error.` ended the turn instead of being
    retried, and one blip cost 37 of 47 eval cases in a live run. `_TRANSIENT`
    has always listed "connection"; nothing was ever asking it.
    """
    from retail_agent.config import Settings
    from retail_agent.llm.provider import build_llm
    from retail_agent.llm.resilience import ResilientChatModel

    _openrouter_only(monkeypatch)

    model = build_llm(Settings(_env_file=None))

    assert isinstance(model, ResilientChatModel)


def test_a_single_provider_chain_is_still_one_provider(monkeypatch):
    """Wrapping must not invent a fallback. Retry the provider you have; do not
    quietly send an executive's question somewhere they did not configure."""
    from retail_agent.config import Settings
    from retail_agent.llm.provider import build_chain

    _openrouter_only(monkeypatch)

    assert [name for name, _ in build_chain(Settings(_env_file=None))] == ["openrouter"]
