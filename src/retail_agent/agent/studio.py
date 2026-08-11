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


def build_studio_graph(*, llm=None, source=None):
    """`llm` and `source` are injectable so this path can be tested without
    credentials — the drift that broke Studio was invisible precisely because
    nothing could construct it in a test.

    No `user_id`/`session_id` parameters: identity is not a property of how
    this graph is built, it is a property of how it is *run*. This function
    never invokes the graph it returns, so there is no `runtime.context` here
    to put anything on. Whoever calls `.invoke`/`.ainvoke` against the graph
    this returns must supply identity there — see `make_graph` for the two
    routes that actually reach it — or every identity-scoped tool call fails
    loudly with `MissingTurnIdentity` (`middleware.py`) rather than silently
    acting as an empty-string user.
    """
    return build_agent(
        _deps(llm=llm, source=source),
        TurnCapture(),
    )


def make_graph(config):
    """The name `langgraph.json` points at. Called once per run.

    One parameter, named `config`: `_factory_utils.py:91-140` classifies a
    factory by its signature, and a second parameter would be read as a
    `ServerRuntime`. Left unused otherwise — this used to resolve `user_id`
    and `thread_id` out of `config["configurable"]` and thread them into
    `build_studio_graph`, but no tool has read anything off that path since
    identity moved to `runtime.context`; the resolution was dead code feeding
    parameters nothing consumed. Both are gone now.

    Tool-facing identity, and the trace label, are a property of the *run*,
    not of how this graph was built: every identity-scoped tool, and the
    recorder middleware that writes the trace, read `runtime.context.user_id`
    — set by whatever the caller passes as `context` (or has synced into
    `context` for it — see below) on the `invoke`/`ainvoke` call made against
    the graph this returns, never by anything resolved in this function.
    `_identity_guard` (`middleware.py`, a `wrap_tool_call` hook so it also
    covers a resumed turn) refuses to run any tool for a turn whose context
    is missing or carries an empty `user_id`, on the first call and on every
    resume alike.

    There is a route through `configurable` that reaches `TurnContext`, and
    it is worth stating precisely because it is easy to get backwards: this
    docstring used to claim a caller "still has to supply one — just not
    through `configurable`", which is false, verified against the installed
    `langgraph_api`. `models/run.py:229` reads `context = payload.get(
    "context") or {}`; when that is empty, `:257-261` falls back to
    `context = configurable.copy()`, and `stream.py:193-201` filters that
    against the graph's declared `context_schema` before the run starts. So
    posting `{"configurable": {"user_id": "..."}}` with no top-level
    `context` in the body DOES reach `TurnContext.user_id`. What does not
    work is supplying both in the same request: `models/run.py:232-236`
    returns 400 if a body carries both `configurable` and `context`, so a
    caller that also needs some other `configurable` key has exactly one
    channel open for identity too — `configurable` — not two.

    No silent default is restored for a caller that supplies neither. A
    Studio run started with no identity now fails loudly with
    `MissingTurnIdentity` on its first tool call — the same failure every
    other caller of this graph gets, and the correct behaviour, not a
    regression to patch around with `"studio"`.
    """
    return build_studio_graph()
