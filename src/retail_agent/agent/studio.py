"""The deployment entrypoint — `langgraph dev`, `langgraph build`, `langgraph up`.

`langgraph_api` invokes a graph factory once per request rather than once at
registration (`langgraph_api/graph.py:390`), handing it that run's config. So
this module exports a factory rather than a graph, and the split between what
is built per process and what is built per run is the whole design:

- `_process_deps` — the model clients, the BigQuery client, the stores. Expensive,
  safe to share, and identical for every run. Built once.
- `make_graph` — the capture. Per run, because it records what one turn did, and
  a shared one silently attributed thread B's reports to thread A.

No checkpointer and no store are attached. The server injects both after this
returns (`graph.py:402`), and supplying our own is a hard `ValueError` under
`langgraph dev` (`graph.py:801`). That constraint outlives this module: nothing
downstream may attach persistence here either.

It does not construct `AgentDeps` itself, for a reason worth keeping: a second
construction site is what broke Studio the last two times a dependency was added
to only one path. `build_deps` is the one place.
"""

from __future__ import annotations

from functools import lru_cache

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.supervisor import build_agent
from retail_agent.bootstrap import build_deps
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.provider import build_models
from retail_agent.obs.tracing import configure_tracing


@lru_cache(maxsize=1)
def _process_deps() -> AgentDeps:
    """Everything expensive, built once for the life of the process.

    Tracing is configured from in here rather than at import, so its os.environ
    write happens on the first request and happens once.

    `lru_cache` only serialises the dict update, not this function body, so
    two concurrent first requests can both miss the cache and both call
    `build_deps` before either result is stored — one of the two built
    `AgentDeps` is then discarded. Harmless here (`AgentDeps` is frozen and
    nothing it holds is mutated), but "built once" is a description of the
    common case, not a guarantee.
    """
    settings = get_settings()
    configure_tracing(settings)
    primary, fallbacks = build_models(settings)
    return build_deps(
        settings,
        llm=primary,
        llm_fallbacks=fallbacks,
        source=BigQuerySource(settings),
    )


def _deps(*, llm=None, source=None) -> AgentDeps:
    """The shared container, or a test's own.

    An injected model is taken alone: a test that supplies its own model is not
    asking for the configured providers to be constructed behind it.
    """
    if llm is None and source is None:
        return _process_deps()

    settings = get_settings()
    configure_tracing(settings)
    primary, fallbacks = (llm, []) if llm is not None else build_models(settings)
    return build_deps(
        settings,
        llm=primary,
        llm_fallbacks=fallbacks,
        source=source if source is not None else BigQuerySource(settings),
    )


def build_studio_graph(
    *, llm=None, source=None, user_id: str = "studio", session_id: str = "studio"
):
    """`llm` and `source` are injectable so this path can be tested without
    credentials — the drift that broke Studio was invisible precisely because
    nothing could construct it in a test."""
    return build_agent(
        _deps(llm=llm, source=source),
        TurnCapture(user_id=user_id, session_id=session_id),
    )


def make_graph(config):
    """The name `langgraph.json` points at. Called once per run.

    One parameter, named `config`: `_factory_utils.py:91-140` classifies a
    factory by its signature, and a second parameter would be read as a
    `ServerRuntime`.

    `user_id` comes from `configurable` rather than from an authenticated
    identity. Real per-user attribution needs the two-parameter `ServerRuntime`
    form and auth configured behind it; neither exists yet, and `"studio"` is
    the honest default until they do.
    """
    configurable = (config or {}).get("configurable", {})
    return build_studio_graph(
        user_id=configurable.get("user_id") or "studio",
        session_id=configurable.get("thread_id") or "studio",
    )
