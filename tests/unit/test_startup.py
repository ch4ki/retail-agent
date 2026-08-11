"""The wiring `retail-agent chat` does at startup.

This exists because it broke in the user's hands: `/persona` was added, its
`build_persona_store` import was never added, `import retail_agent.cli.app`
still succeeded, all 413 tests still passed, and the app died on launch with
NameError — reported as "Could not connect to BigQuery", which sent the user to
fix something that was not broken.

A NameError inside a function is invisible to every test that does not call
that function.
"""

import pytest

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
    for field in ("traces", "personas", "preferences", "reports", "policy"):
        assert getattr(deps, field) is not None, f"{field} was not wired"


def test_the_agent_can_be_built_from_them():
    """The other half: a store that exists but has the wrong shape would pass
    the check above and fail when a turn runs.

    A model that can `bind_tools` is required where the graph needed nothing —
    `create_agent` binds the tool schemas when the agent is compiled.
    """
    from retail_agent.agent.supervisor import build_agent

    deps = build_deps(_settings(), llm=_bindable(), source=_schema_only())
    assert build_agent(deps)


def test_no_tool_leaks_its_runtime_into_the_model_facing_schema():
    """`ToolRuntime` must never reach the model.

    A tool takes it by adding a parameter literally named `runtime` — nothing
    marks it `Annotated` or optional — and the framework strips that parameter
    from `tool_call_schema` and injects a real value at call time. If a tool's
    `runtime` parameter is ever given a default (to make some other caller's
    life easier, say), the strip stops working: `langchain_core`'s injection
    check does not unwrap `Optional`, so `runtime` stays in the schema, and
    schema conversion for every real provider then crashes on `ToolRuntime`'s
    non-serialisable `stream_writer`. The fakes used elsewhere in this suite
    override `bind_tools` and never run real schema conversion, so only a test
    that calls `convert_to_openai_tool` directly, as this one does, can catch
    it.

    Covers every tool the process can bind to a model, not just the ten
    supervisor tools `build_tools` returns. `run_sql` and `lookup_definitions`
    come from `build_analyst_tools` and are bound to the nested analyst
    agent's own model (`subagents.py`'s `analyst` tool), a second `bind_tools`
    call this suite would otherwise never exercise. Reinstating
    `runtime: ToolRuntime | None = None` on `run_sql` alone — the original
    defect from Task 1 — passed the whole suite at 872 with this loop scoped
    to `build_tools` only; only widening the loop here catches it.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from retail_agent.agent.supervisor import build_tools
    from retail_agent.agent.tools import build_analyst_tools

    deps = build_deps(_settings(), llm=_bindable(), source=_schema_only())
    tools = [*build_tools(deps), *build_analyst_tools(deps)]
    for tool in tools:
        assert "runtime" not in tool.tool_call_schema.model_fields, (
            f"{tool.name} leaks `runtime` into its model-facing schema"
        )
        convert_to_openai_tool(tool)  # must not raise


def test_an_unreachable_database_degrades_rather_than_raising():
    deps = build_deps(_settings(), llm=object(), source=object())

    from retail_agent.obs.traces import InMemoryTraceStore
    from retail_agent.store.reports import InMemoryReportStore

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

    assert build_studio_graph(llm=_bindable(), source=_schema_only())


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


# --- the deployment entrypoint ---
#
# `langgraph_api` invokes a graph factory once per request
# (langgraph_api/graph.py:390), passing that run's config. These tests pin the
# three properties that depend on: one dependency container per process, one
# freshly compiled agent per run, and no persistence of our own.


def test_the_factory_takes_exactly_a_config():
    """`langgraph_api/_factory_utils.py:91-140` classifies a factory by its
    signature. A second parameter would be read as a `ServerRuntime` and fail
    to resolve. Checked directly rather than through that private helper."""
    import inspect

    from retail_agent.agent.studio import make_graph

    parameters = list(inspect.signature(make_graph).parameters)

    assert parameters == ["config"]


@pytest.fixture
def _fake_deps(monkeypatch):
    """Point the credentialed path at doubles, so `_process_deps` can run.

    Clears the `_process_deps` cache on both sides: on entry, so a stale entry
    from an earlier test cannot stand in for this one's doubles; on exit, so
    this test does not leave an `AgentDeps` built from a fake model against an
    unreachable database sitting in the cache for whichever test runs next.
    `pytest-randomly` is not installed here, but test order is not otherwise
    guaranteed, so that staleness would still be a live ordering hazard, not
    just untidiness.
    """
    from retail_agent.agent import studio
    from retail_agent.config import Settings

    monkeypatch.setattr(
        studio, "get_settings", lambda: Settings(_env_file=None, database_url=UNREACHABLE)
    )
    monkeypatch.setattr(studio, "configure_tracing", lambda settings: False)
    monkeypatch.setattr(studio, "build_models", lambda settings: (_bindable(), []))
    monkeypatch.setattr(studio, "BigQuerySource", lambda settings: _schema_only())
    studio._process_deps.cache_clear()
    yield studio
    studio._process_deps.cache_clear()


def test_each_run_gets_its_own_compiled_agent(_fake_deps, monkeypatch):
    """The defect this replaces: one long-lived capture shared by every Studio
    thread, so two conversations wrote their events, reports and pending
    approvals into the same object.

    There is no such object to share any more — a turn's record lives in
    checkpointed `TurnState`, and identity lives on `TurnContext`, both set by
    whoever calls `.invoke` against the graph this returns, per run. What
    remains to pin here is that `make_graph` still compiles a fresh agent on
    every call rather than reusing one across runs; object identity of what
    `build_agent` returns is the only thing left to check that on, and it is
    also the only thing the defect it replaces was ever really about.
    """
    studio = _fake_deps

    seen = []
    real_build_agent = studio.build_agent

    def spy(deps, **kw):
        built = real_build_agent(deps, **kw)
        seen.append(built)
        return built

    monkeypatch.setattr(studio, "build_agent", spy)

    studio.make_graph({"configurable": {"thread_id": "thread-a"}})
    studio.make_graph({"configurable": {"thread_id": "thread-b"}})

    assert seen[0] is not seen[1]


def test_the_expensive_dependencies_are_built_once(_fake_deps, monkeypatch):
    """Per-run construction must not mean per-run BigQuery and model clients."""
    studio = _fake_deps
    monkeypatch.setattr(studio, "build_agent", lambda deps, **kw: None)

    studio.make_graph({"configurable": {"thread_id": "one"}})
    studio.make_graph({"configurable": {"thread_id": "two"}})

    info = studio._process_deps.cache_info()
    assert (info.misses, info.hits) == (1, 1)


def test_the_returned_graph_carries_no_persistence(_fake_deps):
    """`langgraph_api/graph.py:801-822` raises under `langgraph dev` if the
    graph has a checkpointer or a store; the server injects its own at
    graph.py:402. The assertion mirrors that isinstance check exactly."""
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

    studio = _fake_deps

    compiled = studio.make_graph({"configurable": {"thread_id": "t"}})

    assert not isinstance(compiled.checkpointer, BaseCheckpointSaver)
    assert not isinstance(compiled.store, BaseStore)


def _bindable():
    """A model `create_agent` will accept.

    `object()` was enough for the graph, whose nodes only ever called `invoke`.
    Compiling an agent binds the tool schemas, so the double has to answer
    `bind_tools`.
    """
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    class Bindable(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return Bindable(responses=[])


def _schema_only():
    """A source that can describe itself and nothing else.

    The analyst renders the schema into its prompt, and Studio compiles at
    import time, so a source that cannot answer `describe_all` fails
    `langgraph dev` at load — which is the failure this file exists to catch.
    """
    from retail_agent.datasources.base import ColumnSchema, TableSchema

    class SchemaOnly:
        def describe_all(self):
            return [
                TableSchema(
                    name="order_items",
                    columns=(ColumnSchema("sale_price", "FLOAT"),),
                )
            ]

    return SchemaOnly()


# --- `retail-agent trios` ---
#
# The subcommand exists because editing `seeds.py` does nothing to a database
# that has already been seeded, and nothing in the interface said so.


def _trios_cli(argv, store):
    """Run the subcommand against a store, capturing what it printed."""
    import io

    from rich.console import Console

    from retail_agent.cli.app import run_trios

    console = Console(record=True, width=100, file=io.StringIO())
    code = run_trios(argv, console=console, store=store)
    return code, console.export_text()


def test_reporting_drift_changes_nothing_without_force():
    from dataclasses import replace

    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.knowledge.trios import InMemoryTrioStore

    stale = replace(SEED_TRIOS[0], metric_definitions={"churn": "older"})
    store = InMemoryTrioStore([stale, *SEED_TRIOS[1:]])

    code, text = _trios_cli([], store)

    assert code == 0
    assert SEED_TRIOS[0].id in text
    assert "--force" in text, "the reader is told how to apply it"
    assert store.get(SEED_TRIOS[0].id) == stale, "nothing was written"


def test_force_rewrites_the_drifted_trio():
    from dataclasses import replace

    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.knowledge.trios import InMemoryTrioStore

    stale = replace(SEED_TRIOS[0], metric_definitions={"churn": "older"})
    store = InMemoryTrioStore([stale, *SEED_TRIOS[1:]])

    code, _ = _trios_cli(["--force"], store)

    assert code == 0
    assert store.get(SEED_TRIOS[0].id) == SEED_TRIOS[0]


def test_an_up_to_date_corpus_says_so():
    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.knowledge.trios import InMemoryTrioStore

    code, text = _trios_cli([], InMemoryTrioStore(SEED_TRIOS))

    assert code == 0
    assert "up to date" in text.lower()


def test_only_the_analyst_is_an_agent():
    """An agent loop with no tools can only ever make one model call, and
    `create_agent` compiles a graph — inside the tool body, so on every
    invocation — to do what one `model.invoke` does.

    Checked across every module under `retail_agent.agent`, not just
    `subagents`, because the property is "nobody added another one anywhere
    in the package" — a `create_agent` added to `reports.py`, `memory.py`, or
    a new module would be just as much a tool-less capability wrongly
    compiled as a graph, and a check scoped to one file cannot see it.

    `supervisor.py` is excluded: its one `create_agent` call is the top-level
    graph itself, the single real agent the whole product is built from
    (`supervisor.py`'s own module docstring says as much), not a subagent —
    counting it would make "exactly one" impossible from the day this test is
    written. No behavioural test can see the property checked here either
    way. Same technique as `test_both_entry_points_use_the_same_wiring`
    above.
    """
    from pathlib import Path

    from retail_agent import agent as agent_package

    package_dir = Path(agent_package.__file__).parent
    counts = {}
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "supervisor.py":
            continue
        count = path.read_text().count("create_agent(")
        if count:
            counts[path.name] = count

    total = sum(counts.values())
    assert total == 1, (
        "exactly one subagent should compile a graph — the analyst, which has "
        "real tools. A tool-less capability belongs in a plain model call "
        f"through `resilient_call`. Found {total} call(s) across: {counts}"
    )
