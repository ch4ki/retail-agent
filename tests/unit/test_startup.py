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
from retail_agent.bootstrap import build_deps
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


# --- the other entry point ---


def test_the_studio_graph_builds():
    """LangGraph Studio loads `studio.py` at import time and had its own copy
    of the wiring. When `traces` was added to the CLI's construction and not to
    that copy, `langgraph dev` failed to load and no test noticed — the module
    could not be built without credentials, so nothing tried.

    It goes through `build_deps` now, and this builds it with the credentialed
    parts injected.
    """
    from retail_agent.agent.studio import build_studio_graph

    assert build_studio_graph(llm=object(), source=object())


def test_both_entry_points_use_the_same_wiring():
    """The guarantee is structural: one function, two callers. A second
    construction site is what broke Studio, so this asserts there is not one."""
    import inspect

    from retail_agent.agent import studio
    from retail_agent.cli import chat

    for module in (studio, chat):
        source = inspect.getsource(module)
        assert "AgentDeps(" not in source, (
            f"{module.__name__} constructs AgentDeps directly; it should call "
            f"build_deps so a new dependency cannot be added to one path only"
        )
        assert "build_deps" in source


# --- the ReAct arm's entry point ---


def test_the_react_studio_graph_builds():
    """Same lesson as `test_the_studio_graph_builds`: Studio imports this at
    load time, so a name that does not resolve fails `langgraph dev` and no
    other test would notice.

    Unlike the graph's, this needs a source carrying a schema: the ReAct arm
    renders the schema into its system prompt when the agent is *built*, where
    the graph's nodes read it per invocation. So `langgraph dev` describes the
    tables once at load rather than on the first question.
    """
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    from retail_agent.datasources.base import ColumnSchema, TableSchema

    class Bindable(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class SchemaOnly:
        def describe_all(self):
            return [
                TableSchema(
                    name="order_items",
                    columns=(ColumnSchema("sale_price", "FLOAT"),),
                )
            ]

    from retail_agent.baseline.studio import build_react_studio_graph

    assert build_react_studio_graph(llm=Bindable(responses=[]), source=SchemaOnly())


def test_the_react_entry_point_shares_the_same_wiring():
    """One `build_deps`, three callers now. A second construction site is what
    broke Studio the first time."""
    import inspect

    from retail_agent.baseline import studio as react_studio

    source = inspect.getsource(react_studio)
    assert "AgentDeps(" not in source
    assert "build_deps" in source


def test_studio_serves_both_arms():
    """The point of registering it: seeing the two graphs side by side is the
    clearest statement of what the comparison is about."""
    import json
    from pathlib import Path

    config = json.loads(Path("langgraph.json").read_text())

    assert set(config["graphs"]) == {"retail_agent", "react_baseline"}
