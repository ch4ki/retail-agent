"""Everything a node needs, injected once rather than imported ad hoc.

Keeping this explicit is what lets component tests swap in a fake LLM and a
fake warehouse without patching module globals.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from retail_agent.config import Settings
from retail_agent.datasources.base import DataSource
from retail_agent.safety.pii import PiiPolicy
from retail_agent.obs.traces import TraceStore
from retail_agent.store.learning import SignalStore
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
    signals: SignalStore | None = None
