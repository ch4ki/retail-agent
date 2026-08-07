"""Graph entrypoint for the ReAct arm in LangGraph Studio (`langgraph dev`).

Registered alongside `retail_agent` so the two arms can be opened side by side.
That view is the clearest statement of what this comparison is about: one graph
where the safety properties are edges you can point at, and one where the model
picks the next step and the middleware bounds it.

Mirrors `agent/studio.py` — same `build_deps`, no checkpointer, because Studio
owns thread persistence. It does not construct `AgentDeps` itself, for the
reason recorded there: a second construction site is what broke Studio the last
time a dependency was added to only one path.
"""

from __future__ import annotations

from retail_agent.baseline.capture import ResultCapture
from retail_agent.baseline.react import build_react_agent
from retail_agent.bootstrap import build_deps
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.provider import build_llm
from retail_agent.obs.tracing import configure_tracing


def build_react_studio_graph(*, llm=None, source=None):
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
    # One capture for the process rather than one per turn. Studio is for
    # looking at the graph, not for scoring it — `ask_once` builds a fresh
    # capture per turn, and that is the path the eval uses.
    return build_react_agent(deps, ResultCapture())


graph = build_react_studio_graph()
