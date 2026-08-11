"""Everything the tools need, injected once rather than imported ad hoc.

Keeping this explicit is what lets component tests swap in a fake LLM and a
fake warehouse without patching module globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel

from retail_agent.config import Settings
from retail_agent.datasources.base import DataSource
from retail_agent.safety.pii import PiiPolicy
from retail_agent.obs.traces import TraceStore
from retail_agent.knowledge.trios import Trio, TrioStore
from retail_agent.store.definitions import DefinitionStore
from retail_agent.store.personas import PersonaStore
from retail_agent.store.preferences import PreferenceStore
from retail_agent.store.reports import ReportStore


@dataclass(frozen=True)
class AgentDeps:
    settings: Settings
    llm: BaseChatModel
    source: DataSource
    policy: PiiPolicy
    reports: ReportStore
    traces: TraceStore
    # The rest of the provider chain, in order. Empty is the ordinary single-
    # provider deployment: `ModelFallbackMiddleware` is only added when this is
    # non-empty, so nothing wraps the model for a fallback that does not exist.
    llm_fallbacks: list[BaseChatModel] = field(default_factory=list)
    personas: PersonaStore | None = None
    preferences: PreferenceStore | None = None
    definitions: DefinitionStore | None = None
    # The Golden Bucket. Empty is a valid state — the undefined-term rule is
    # what protects the answer when nothing is retrieved.
    # A `TrioStore`, or a plain list in tests. `live_trios` accepts either.
    trios: TrioStore | list[Trio] = field(default_factory=list)
    # Optional second ranker for retrieval. None means lexical only, which is
    # the default and needs no model and no provider.
    dense: object | None = None


@dataclass(frozen=True)
class TurnContext:
    """Who is asking, and which turn this is. Supplied per run.

    Three strings and nothing else. On the LangGraph server this arrives from
    the API request body, so anything unserialisable here would not survive the
    trip — never put `deps`, a store or a client on it.

    Identity lives here rather than in checkpointed graph state because it is
    fixed for a run and set by the caller: never accumulated, never merged,
    never needing a reducer. That is exactly what runtime context is for, and
    it is what lets one compiled graph serve two users.
    """

    user_id: str = ""
    session_id: str = ""
    turn_id: str = ""
