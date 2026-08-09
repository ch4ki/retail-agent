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
