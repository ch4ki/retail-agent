"""Single source of truth for configuration. Everything env-derived lives here."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "openai", "openrouter", "ollama"]

DEFAULT_MODELS: dict[Provider, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "openrouter": "google/gemini-2.5-flash",
    "ollama": "llama3.1:8b",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    llm_provider: Provider = "gemini"
    # Applies to whichever provider is active. Prefer the per-provider vars
    # below when you keep more than one configured.
    llm_model: str | None = None
    gemini_model: str | None = None
    openai_model: str | None = None
    openrouter_model: str | None = None
    ollama_model: str | None = None
    llm_temperature: float = 0.0
    # Explicit output cap. Without one, providers reserve credit for the model's
    # full output ceiling (OpenRouter 402s on this), and per-turn cost is
    # unbounded. Every prompt here wants a query or a few paragraphs.
    llm_max_tokens: int = 2048
    google_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Providers to fall back to, in order, when the primary keeps failing.
    # Comma-separated; ones without credentials are dropped rather than fatal.
    # Empty means no chain, which is the common single-provider case.
    llm_fallbacks: str = ""
    llm_retry_attempts: int = 3
    llm_breaker_threshold: int = 3
    llm_breaker_cooldown_seconds: float = 60.0

    # --- BigQuery ---
    google_cloud_project: str | None = None
    bq_dataset: str = "bigquery-public-data.thelook_ecommerce"
    bq_max_bytes_billed: int = 2_000_000_000  # 2 GB per query
    bq_timeout_seconds: int = 60
    allowed_tables: frozenset[str] = frozenset(
        {
            "orders",
            "order_items",
            "products",
            "users",
            "inventory_items",
            "distribution_centers",
        }
    )

    # --- Query shaping ---
    default_row_limit: int = 500
    max_row_limit: int = 5_000

    # --- Storage ---
    database_url: str = "postgresql://retail:retail@localhost:5433/retail_agent"

    # --- Observability ---
    # Needs both the flag and the key; see obs.tracing.configure_tracing.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "retail-agent"
    langsmith_endpoint: str | None = None

    # --- Knowledge ---
    # Dense retrieval over the trio corpus, via embedded Milvus Lite with a
    # local ONNX embedding model. Off by default: the first call downloads the
    # model, and lexical retrieval already works with no provider at all.
    dense_retrieval: bool = False
    milvus_path: str = "./.milvus/trios.db"
    # Which embedder ranks the corpus. `openai` when a key is configured,
    # because the bundled local model measurably cannot do this job: on the seed
    # corpus its scores for relevant questions (0.138-0.517) overlap its scores
    # for nonsense (up to 0.222), so no relevance floor separates them.
    # `text-embedding-3-small` scores relevant 0.296+ and nonsense below 0.102.
    # Set `local` to keep everything on the machine and accept the weaker
    # ranking; see `docs/design.md` §5.1.
    embedding_backend: Literal["auto", "openai", "local"] = "auto"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Safety ---
    pii_salt: str = "dev-salt-change-me"

    # --- Agent ---
    # Failures tolerated per turn, not per step. The turn ends on the failure
    # that empties it, so this is the attempt count: 3 attempts, 2 retries.
    repair_budget: int = 3
    # Empty results get their own budget. An empty result is not necessarily a
    # bug — sometimes "no orders matched" is the true answer — so spending the
    # syntax budget on it would starve genuinely broken SQL.
    diagnose_budget: int = 1
    max_analysis_steps: int = 10
    # Prior messages shown to the router and planner so that "compare that to
    # April" resolves to something queryable. Bounded: history grows for a whole
    # session, and every turn pays for it in tokens.
    history_messages: int = 6

    @property
    def resolved_model(self) -> str:
        """Model for the active provider."""
        return self.model_for(self.llm_provider)

    def model_for(self, provider: str) -> str:
        """Model for any provider in the chain.

        A name pinned for one provider must never be sent to another, so the
        per-provider variable wins over the generic one — and the generic
        `LLM_MODEL` applies only to the *active* provider. Without that second
        rule, setting `LLM_MODEL=gemini-2.5-pro` would send "gemini-2.5-pro" to
        the OpenAI fallback the moment Gemini went down.
        """
        specific = getattr(self, f"{provider}_model", None)
        if specific:
            return specific
        if provider == self.llm_provider and self.llm_model:
            return self.llm_model
        return DEFAULT_MODELS[provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
