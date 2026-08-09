"""Graph entrypoint for LangGraph Studio (`langgraph dev`).

Studio owns thread persistence, so no checkpointer is passed here. It does not
construct `AgentDeps` itself for a reason worth keeping: a second construction
site is what broke Studio the last two times a dependency was added to only one
path. `build_deps` is the one place.

The capture is per-process rather than per-turn. Studio is for looking at the
agent, not for scoring it — `ask_once` builds a fresh capture per turn, and that
is the path the eval and the CLI use.
"""

from __future__ import annotations

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.supervisor import build_agent
from retail_agent.bootstrap import build_deps
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.provider import build_models
from retail_agent.obs.tracing import configure_tracing


def build_studio_graph(*, llm=None, source=None):
    """`llm` and `source` are injectable so this path can be tested without
    credentials — the drift that broke Studio was invisible precisely because
    nothing could construct it in a test."""
    settings = get_settings()
    configure_tracing(settings)

    # An injected model is taken alone: a test that supplies its own model is
    # not asking for the configured providers to be constructed behind it.
    primary, fallbacks = (llm, []) if llm is not None else build_models(settings)

    deps = build_deps(
        settings,
        llm=primary,
        llm_fallbacks=fallbacks,
        source=source if source is not None else BigQuerySource(settings),
    )
    return build_agent(deps, TurnCapture(user_id="studio", session_id="studio"))


graph = build_studio_graph()
