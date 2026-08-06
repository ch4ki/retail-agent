"""Graph entrypoint for LangGraph Studio (`langgraph dev`).

Studio needs a compiled graph as a module-level object, and it supplies its own
persistence — so unlike the CLI this builds the graph with no checkpointer.

Everything else is the same wiring the CLI uses, through the same function. It
used to be a copy of it, described in this docstring as "not a parallel code
path that could drift", and it drifted: `traces` was added to the CLI's
construction and not to this one, and Studio failed to load.
"""

from __future__ import annotations

from retail_agent.agent.graph import build_graph
from retail_agent.bootstrap import build_deps
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.provider import build_llm
from retail_agent.obs.tracing import configure_tracing


def build_studio_graph(*, llm=None, source=None):
    """`llm` and `source` are injectable so this path can be tested without
    credentials — the drift that broke Studio was invisible precisely because
    nothing could construct it in a test."""
    settings = get_settings()
    configure_tracing(settings)

    deps = build_deps(
        settings,
        llm=llm if llm is not None else build_llm(settings),
        source=source if source is not None else BigQuerySource(settings),
    )
    # No checkpointer: the Studio server owns thread persistence.
    return build_graph(deps)


graph = build_studio_graph()
