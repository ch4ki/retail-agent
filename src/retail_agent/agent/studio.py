"""The deployment entrypoint — `langgraph dev`, `langgraph build`, `langgraph up`.

Upstream line numbers below were re-verified against langgraph-api 0.12.3; they
drift on upgrade, so treat a mismatch as stale documentation, not as a finding.

`langgraph_api` invokes a graph factory once per request rather than once at
registration (`langgraph_api/graph.py:397`), handing it that run's config. So
this module exports a factory rather than a graph, and the split between what
is built per process and what is built per run is the whole design:

- `_process_deps` — the model clients, the BigQuery client, the stores. Expensive,
  safe to share, and identical for every run. Built once.
- `make_graph` — the compiled agent. Built per run, because identity and the
  turn's own record are per-run properties now (`TurnContext`, checkpointed
  `TurnState`), not something a shared object could attribute to the wrong
  thread.

No checkpointer and no store are attached. The server supplies both itself, by
writing them into the `configurable` of the very config it then hands this
factory (`graph.py:377-385`, immediately before the `invoke_factory` call), and
supplying our own is a hard `ValueError` under `langgraph dev`
(`graph.py:808-828`). That constraint outlives this module: nothing
downstream may attach persistence here either.

It does not construct `AgentDeps` itself, for a reason worth keeping: a second
construction site is what broke Studio the last two times a dependency was added
to only one path. `build_deps` is the one place.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import count

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.supervisor import build_agent
from retail_agent.bootstrap import build_deps
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.provider import build_models
from retail_agent.obs.tracing import configure_tracing


def _build_process_deps() -> AgentDeps:
    """The construction itself. Blocking, by way of the BigQuery client.

    Tracing is configured from in here rather than at import, so its os.environ
    write happens on the first request and happens once.
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


_pool_ids = count()


@lru_cache(maxsize=1)
def _process_deps() -> AgentDeps:
    """Everything expensive, built once for the life of the process.

    Built on a worker thread whenever an event loop is running, and directly
    otherwise. `make_graph` is called per request by `langgraph_api`, which
    calls it from the loop — and constructing the BigQuery client resolves
    Google's application default credentials, which for authorized-user
    credentials (`gcloud auth application-default login`) shells out to
    `gcloud config get project`: `google/auth/_default.py` falls through to
    `_cloud_sdk.get_project_id()` whenever the credentials file carries no
    project of its own, which an authorized-user file never does.

    A blocking subprocess on the loop thread is exactly what `langgraph dev`'s
    BlockBuster exists to catch, and it caught it — `BlockingError: Blocking
    call to os.read` on the first run. BlockBuster only raises when
    `asyncio.get_running_loop()` succeeds (`blockbuster.py:77-79`), so a worker
    thread is enough; the loop still waits for the result, but it waits once
    per process rather than dying.

    `lru_cache` only serialises the dict update, not this function body, so
    two concurrent first requests can both miss the cache and both call
    `_build_process_deps` before either result is stored — one of the two built
    `AgentDeps` is then discarded. The discarded object itself is harmless
    (`AgentDeps` is frozen and nothing it holds is mutated), but building it is
    not free of side effects: `build_deps` → `build_trio_store` seeds the trio
    table with a read-then-insert against a primary-keyed table, with no
    upsert (`knowledge/trios.py:472`, `store/models.py:226`). Two builds racing
    against a fresh database can both see a trio absent and both INSERT,
    raising `IntegrityError`. That race is pre-existing and unchanged by this
    fix — not addressed here — but "built once" is a description of the common
    case, not a guarantee, and this is what the uncommon case can do.

    `future.result()` below, and the pool's implicit `shutdown(wait=True)` on
    `__exit__`, both block the loop thread on a `Thread.join()` — itself a
    blocking lock acquisition BlockBuster would normally catch just as it
    caught the subprocess call above. This works only because langgraph's dev
    server deactivates `threading.Lock.acquire` for BlockBuster's purposes
    (`langgraph_runtime_inmem/queue.py:346`, reasoning that anything using a
    thread pool would be relying on it). Tightening that exemption anywhere
    upstream would break the approach taken here.

    Zero loop stall is possible in principle: `langgraph_api/asyncio.py:30-31`
    awaits a coroutine returned by the factory, and `_factory_utils.py`'s
    `_classify_factory` classifies by signature alone, so an `async def
    make_graph(config)` that itself does `await asyncio.to_thread(...)` would
    never block the loop at all. Not taken here because it changes
    `make_graph`'s signature and needs its own `langgraph dev` verification —
    left as a follow-up, not folded into this fix.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop: the CLI and the eval. Nothing to protect, so no thread.
        return _build_process_deps()

    # A unique prefix per pool: two concurrent builds would otherwise both
    # name their single worker "deps_0", and that name is the evidence relied
    # on elsewhere that a build went off-loop.
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"deps-{next(_pool_ids)}"
    ) as pool:
        return pool.submit(_build_process_deps).result()


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
    return build_agent(_deps(llm=llm, source=source))


def make_graph(config):
    """The name `langgraph.json` points at. Called once per run.

    One parameter, named `config`: `_factory_utils.py:92-142` classifies a
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
