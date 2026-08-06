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

    # --- Safety ---
    pii_salt: str = "dev-salt-change-me"

    # --- Agent ---
    # Failures tolerated per turn, not per step. The turn ends on the failure
    # that empties it, so this is the attempt count: 3 attempts, 2 retries.
    repair_budget: int = 3
    max_analysis_steps: int = 10
    # Prior messages shown to the router and planner so that "compare that to
    # April" resolves to something queryable. Bounded: history grows for a whole
    # session, and every turn pays for it in tokens.
    history_messages: int = 6

    @property
    def resolved_model(self) -> str:
        """Model for the active provider.

        A model name pinned for one provider must never be sent to another, so
        the per-provider variable wins over the generic one.
        """
        specific = getattr(self, f"{self.llm_provider}_model", None)
        return specific or self.llm_model or DEFAULT_MODELS[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
