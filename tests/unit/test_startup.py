"""The wiring `retail-agent chat` does at startup.

This exists because it broke in the user's hands: `/persona` was added, its
`build_persona_store` import was never added, `import retail_agent.cli.app`
still succeeded, all 413 tests still passed, and the app died on launch with
NameError — reported as "Could not connect to BigQuery", which sent the user to
fix something that was not broken.

A NameError inside a function is invisible to every test that does not call
that function.
"""

from retail_agent.agent.deps import AgentDeps
from retail_agent.cli.app import build_deps
from retail_agent.config import Settings

# Nothing listening here, so every store takes its degraded path.
UNREACHABLE = "postgresql://nobody@127.0.0.1:1/nothing"


def _settings():
    return Settings(_env_file=None, database_url=UNREACHABLE)


def test_every_dependency_is_wired():
    """Names resolve, stores construct, and the graph gets a complete
    container — without credentials, a database, or a network."""
    deps = build_deps(_settings(), llm=object(), source=object())

    assert isinstance(deps, AgentDeps)
    for field in ("traces", "personas", "preferences", "signals", "reports", "policy"):
        assert getattr(deps, field) is not None, f"{field} was not wired"


def test_the_graph_can_be_built_from_them():
    """The other half: a store that exists but has the wrong shape would pass
    the check above and fail when the graph runs."""
    from retail_agent.agent.graph import build_graph

    assert build_graph(build_deps(_settings(), llm=object(), source=object()))


def test_an_unreachable_database_degrades_rather_than_raising():
    deps = build_deps(_settings(), llm=object(), source=object())

    from retail_agent.obs.traces import InMemoryTraceStore
    from retail_agent.store.memory_reports import InMemoryReportStore

    assert isinstance(deps.traces, InMemoryTraceStore)
    assert isinstance(deps.reports, InMemoryReportStore)


def test_a_persona_is_available_even_with_no_database():
    """The prompt slot must never be empty, whatever the database is doing."""
    from retail_agent.store.personas import active_body

    deps = build_deps(_settings(), llm=object(), source=object())

    assert active_body(deps.personas).strip()


def test_degradation_is_reported_to_the_user():
    class Recorder:
        def __init__(self):
            self.messages = []

        def print(self, message):
            self.messages.append(str(message))

    console = Recorder()
    build_deps(_settings(), llm=object(), source=object(), console=console)

    assert any("Postgres is unreachable" in m for m in console.messages)
