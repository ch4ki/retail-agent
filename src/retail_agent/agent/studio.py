"""Graph entrypoint for LangGraph Studio (`langgraph dev`).

Studio needs a compiled graph as a module-level object, and it supplies its own
persistence — so unlike the CLI this builds the graph with no checkpointer.

Everything else is the same wiring the CLI uses, deliberately: a Studio run
exercises the real guard, the real masking policy and the real warehouse, not a
parallel code path that could drift.
"""

from __future__ import annotations

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.graph import build_graph
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.provider import build_llm
from retail_agent.obs.tracing import configure_tracing
from retail_agent.safety.pii import PiiPolicy


def build_studio_graph():
    settings = get_settings()
    configure_tracing(settings)

    deps = AgentDeps(
        settings=settings,
        llm=build_llm(settings),
        source=BigQuerySource(settings),
        policy=PiiPolicy.default(),
    )
    # No checkpointer: the Studio server owns thread persistence.
    return build_graph(deps)


graph = build_studio_graph()
