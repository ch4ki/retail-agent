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
    # How many rows are fetched and shown. Applied when reading the result, not
    # as a LIMIT in the SQL: a LIMIT truncates server-side, so the true size of
    # the result is lost and an aggregate meant to be counted comes back capped.
    # BigQuery bills on bytes scanned and a LIMIT saves none of them — measured
    # at 0% on the query that exposed this — so it was never buying cost.
    display_row_limit: int = 500
    # The SQL-level bound the guard still injects: a safety ceiling against an
    # unbounded result, not a display cap. High enough that any realistic
    # analytical result is complete, so `total_rows` is the real count.
    max_row_limit: int = 100_000

    # --- Storage ---
    database_url: str = "postgresql://retail:retail@localhost:5433/retail_agent"

    # --- Observability ---
    # Needs both the flag and the key; see obs.tracing.configure_tracing.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "retail-agent"
    langsmith_endpoint: str | None = None

    # --- Knowledge ---
    # Dense retrieval over the trio corpus, stored as pgvector columns in the
    # database that already holds the trios. Off by default: lexical retrieval
    # works with no key and no database at all.
    dense_retrieval: bool = False
    # Vectors live in Postgres via pgvector, so dense retrieval needs the
    # database as well as an embedding key. Without either it degrades to
    # lexical rather than failing.
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
