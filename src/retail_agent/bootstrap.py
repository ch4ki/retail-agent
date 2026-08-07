"""Assembling the agent's dependencies, in one place.

There are two entry points — the CLI and the LangGraph Studio module — and they
must not each wire this up themselves. They did, and it broke twice: a store
added to one construction site and not the other, discovered when a user ran
the app rather than when the tests ran.

One function, called from both, covered by `tests/unit/test_startup.py`.
"""

from __future__ import annotations

from retail_agent.agent.deps import AgentDeps
from retail_agent.knowledge.dense import build_dense_index
from retail_agent.knowledge.trios import build_trio_store
from retail_agent.obs.traces import build_trace_store
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.definitions import build_definition_store
from retail_agent.store.learning import build_signal_store
from retail_agent.store.personas import build_persona_store
from retail_agent.store.preferences import build_preference_store
from retail_agent.store.reports import build_report_store


def build_deps(settings, *, llm, source, console=None) -> AgentDeps:
    """Assemble everything the graph needs from the stores.

    Split out of `_chat` so it can be exercised without credentials — a missing
    name in here used to surface only when a user ran the app, and was reported
    as a BigQuery problem.
    """

    def warn(message: str):
        return (lambda: console.print(f"[yellow]{message}[/yellow]")) if console else None

    trios = build_trio_store(settings)
    # Dense retrieval keeps its vectors beside the trios, so it needs the same
    # database. `None` when Postgres is unreachable, which degrades retrieval to
    # lexical rather than failing the turn.
    sessions = getattr(trios, "sessions", None)

    return AgentDeps(
        settings=settings,
        llm=llm,
        source=source,
        policy=PiiPolicy.default(),
        # Rows, seeded from the hand-authored corpus on first run, so a
        # definition can be edited or superseded without a deploy.
        trios=trios,
        dense=build_dense_index(settings, sessions=sessions),
        traces=build_trace_store(settings),
        personas=build_persona_store(settings),
        preferences=build_preference_store(settings),
        # Postgres-backed: the proposal threshold is three, and evidence that
        # died with the process could only ever be met inside one session.
        signals=build_signal_store(settings),
        definitions=build_definition_store(settings),
        reports=build_report_store(
            settings,
            on_degraded=warn(
                "Reports will not be saved — Postgres is unreachable. Run "
                "`docker compose up -d postgres && uv run retail-agent migrate`."
            ),
        ),
    )
