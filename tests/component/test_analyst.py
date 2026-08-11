"""The analyst subagent: what it is given, and what it discloses.

`recall_node` ran unconditionally on every analyze turn, and that still holds:
retrieval happens in this wrapper, before the subagent is built, so a model
that elected not to look a term up cannot lose the corpus.

What this no longer does is *decide* whether a term is unsettled. That was a
regex over nineteen words, and it is gone — `ask_for_definitions` asks and
records, and the analyst reads what it recorded. The disclosure is still forced
here rather than requested in a prompt, which is the property worth keeping.
"""

import pandas as pd

from retail_agent.agent.subagents import build_subagents
from retail_agent.knowledge.trios import Trio
from retail_agent.store.definitions import InMemoryDefinitionStore

from .conftest import FakeSource

LOYAL = Trio(
    id="trio-loyal",
    question="How many loyal customers do we have?",
    sql="SELECT 1",
    report="Loyalty is measured over a rolling year.",
    metric_definitions={"loyal": "three or more completed orders in 12 months"},
)


def _text(result):
    """A tool's answer, unwrapped from the `Command` it now returns — the one
    `ToolMessage` on `update["messages"]` carries the text a model would see."""
    from langgraph.types import Command

    if isinstance(result, Command):
        return result.update["messages"][0].content
    return result


def subagents_for(deps):
    return {t.name: t.func for t in build_subagents(deps)}


def _runtime(state=None):
    """A `ToolRuntime` good enough to call a tool's raw `.func` directly.

    The framework only injects a real one when a tool runs through
    `agent.invoke`; calling `.func` here bypasses that machinery entirely, so
    the test has to build one itself. Six of `ToolRuntime`'s nine fields are
    required — `tools`, `execution_info` and `server_info` have defaults.

    `context` carries the same `user_id`/`session_id` used elsewhere in this
    file's fixtures, so a tool reading `runtime.context.user_id` sees the same
    executive throughout. `state` stands in for whatever an earlier tool call
    this turn would have written into `TurnState` — `analyst` reads
    `assumed_terms` off it, and `report_writer` reads `trio_ids`.
    """
    from langchain.tools import ToolRuntime
    from retail_agent.agent.deps import TurnContext

    return ToolRuntime(
        state=state or {},
        context=TurnContext(user_id="exec", session_id="s1"),
        config={},
        stream_writer=None,
        tool_call_id="test",
        store=None,
    )


def test_the_executives_own_definitions_reach_the_model(make_deps):
    """All of them, not the ones a regex picked out of the question. The
    filtering step that needed a term list is gone, and the whole set costs one
    read — so a word the old detector never recognised is now in front of the
    model anyway."""
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="exec", term="LGB", definition="low gross basket")
    source = FakeSource(frames={"default": pd.DataFrame({"n": [9]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "9."],
        src=source,
        definitions=definitions,
    )
    analyst = subagents_for(deps)

    analyst["analyst"]("how many LGB customers?", runtime=_runtime())

    assert any("low gross basket" in prompt for prompt in deps.llm.prompts)


def test_a_trio_settles_the_term_and_the_query_runs(make_deps):
    """With the corpus covering it, nothing is asked and the analysis proceeds."""
    source = FakeSource(frames={"default": pd.DataFrame({"loyal_customers": [42]})})
    deps = make_deps(
        script=[
            [("run_sql", {"sql": "SELECT COUNT(*) AS loyal_customers FROM users"})],
            "42 customers are loyal.",
        ],
        src=source,
        definitions=InMemoryDefinitionStore(),
        trios=[LOYAL],
    )
    analyst = subagents_for(deps)

    result = analyst["analyst"]("who are our loyal customers?", runtime=_runtime())
    answer = _text(result)

    assert "42" in answer
    assert result.update["trio_ids"] == ["trio-loyal"]
    assert source.executed


def test_the_agreed_definition_reaches_the_model(make_deps):
    """Retrieval that never reaches the prompt is retrieval that did nothing."""
    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})})
    deps = make_deps(script=["one."], src=source, trios=[LOYAL])
    analyst = subagents_for(deps)

    analyst["analyst"]("who are our loyal customers?", runtime=_runtime())

    assert any(
        "three or more completed orders" in prompt for prompt in deps.llm.prompts
    )


def test_a_recorded_assumption_is_forced_into_the_answer(make_deps):
    """The number is only trustworthy if the reader knows which judgement made it.

    The note is appended by the wrapper rather than requested in the prompt, so
    a model that ignores the instruction still cannot return the figure alone.
    The terms come from this turn's own state — written earlier in the turn by
    `ask_for_definitions` when nobody was there to answer it.
    """
    source = FakeSource(frames={"default": pd.DataFrame({"n": [9]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "9."],
        src=source,
        definitions=InMemoryDefinitionStore(),
    )
    analyst = subagents_for(deps)

    result = analyst["analyst"](
        "how many LGB customers?", runtime=_runtime(state={"assumed_terms": ["LGB"]})
    )
    answer = _text(result)

    assert "no agreed definition" in answer.lower()
    assert "LGB" in answer


def test_nothing_assumed_means_no_disclosure(make_deps):
    """A caveat on a question that did not need one is noise, and noise is how
    a warning stops being read."""
    source = FakeSource(frames={"default": pd.DataFrame({"n": [3]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "3."],
        src=source,
        definitions=InMemoryDefinitionStore(),
    )
    analyst = subagents_for(deps)

    answer = _text(
        analyst["analyst"](
            "how much revenue did we make in March?", runtime=_runtime()
        )
    )

    assert "3" in answer
    assert "no agreed definition" not in answer.lower()


def test_the_report_writer_is_shown_how_analysts_here_write(make_deps):
    """The other half of what a trio carries, and it had gone missing.

    `metric_definitions` says what to measure and reaches the analyst. `report`
    demonstrates the house shape — split by cohort, compare against a baseline,
    close with numbered actions — which is hard to specify and easy to show. It
    was injected by the graph's synthesis node, and deleting that node dropped
    it silently: nothing failed, the corpus field simply stopped being read.
    """
    deps = make_deps(script=["## Summary\nRevenue rose."], trios=[LOYAL])
    writer = subagents_for(deps)

    writer["report_writer"](
        "Revenue rose 4% in Q1.",
        title="Q1 Revenue",
        runtime=_runtime(state={"trio_ids": ["trio-loyal"]}),
    )

    assert "Loyalty is measured over a rolling year." in deps.llm.prompts[0]


def test_a_report_with_no_trio_behind_it_still_writes(make_deps):
    """An empty corpus is a valid state; the examples block just goes away."""
    deps = make_deps(script=["## Summary\nRevenue rose."])
    writer = subagents_for(deps)

    assert writer["report_writer"]("Revenue rose 4%.", title="Q1 Revenue", runtime=_runtime())


def test_the_report_writer_runs_through_the_provider_chain(make_deps):
    """The tool-less subagent compiles down a different path, and it was broken.

    Every other agent here has tools, so every other agent compiled through
    `bind_tools`. `create_agent` binds a tool-less agent with `bind` instead,
    and the chain object that used to sit in front of the model implemented one
    and not the other — so the first live report died on `AttributeError` while
    the whole offline suite stayed green.

    The chain object is gone and the fallbacks are middleware now, which is
    what makes that class of bug unreachable: middleware is handed a model
    rather than having to impersonate one. Kept, with a fallback configured, so
    the tool-less compile path stays covered.
    """
    deps = make_deps(script=["## Summary\nRevenue rose."])
    object.__setattr__(deps, "llm_fallbacks", [deps.llm])
    writer = subagents_for(deps)

    receipt = _text(
        writer["report_writer"](
            "Revenue rose 4% in Q1.", title="Q1 Revenue", runtime=_runtime()
        )
    )

    # The receipt itself carries no report text (see test_report_tools.py) —
    # what this test needs is proof the tool-less compile path produced a
    # report at all, so it checks the store rather than the return value.
    assert "written" in receipt
    assert "Revenue rose" in deps.reports.list_reports(owner_id="exec")[0].body


def test_the_report_writer_cannot_reach_the_data(make_deps):
    """No tools, so a number missing from the brief cannot appear in the report."""
    deps = make_deps(script=["## Summary\nRevenue rose."])
    writer = subagents_for(deps)

    writer["report_writer"]("Revenue rose 4% in Q1.", title="Q1 Revenue", runtime=_runtime())

    assert "Revenue rose" in deps.reports.list_reports(owner_id="exec")[0].body
    assert deps.llm.bound_tools == []


def test_the_analyst_inherits_the_parent_runs_config(make_deps, monkeypatch):
    """Finding 5. The nested `agent.invoke` used to pass no config, so the
    subagent's model calls reached the parent's callbacks only by contextvar
    propagation — which holds for synchronous Python and silently stops the
    moment anything moves behind a thread or an async boundary. The eval's
    token accounting is what breaks first, and it breaks quietly.

    Contextvar propagation makes the supervisor's own callbacks reach the
    subagent either way, so a test that only counts model-start events cannot
    tell the two implementations apart — deleting `config=runtime.config`
    entirely still leaves it green. What actually distinguishes them is
    whether the nested `agent.invoke` was *handed* the parent's config, so
    this spies on the analyst's inner `create_agent` and inspects the `config`
    kwarg its `invoke` was called with.
    """
    from retail_agent.agent import subagents as subagents_module
    from retail_agent.agent.deps import TurnContext
    from retail_agent.agent.supervisor import build_agent

    captured_configs = []
    real_create_agent = subagents_module.create_agent

    def spying_create_agent(*args, **kwargs):
        nested_agent = real_create_agent(*args, **kwargs)
        real_invoke = nested_agent.invoke

        def spying_invoke(*invoke_args, **invoke_kwargs):
            captured_configs.append(invoke_kwargs.get("config"))
            return real_invoke(*invoke_args, **invoke_kwargs)

        monkeypatch.setattr(nested_agent, "invoke", spying_invoke)
        return nested_agent

    monkeypatch.setattr(subagents_module, "create_agent", spying_create_agent)

    deps = make_deps(
        script=[
            [("analyst", {"question": "how many orders?"})],
            [("run_sql", {"sql": "SELECT COUNT(*) FROM orders"})],
            "Nine.",
            "Nine.",
        ]
    )
    agent = build_agent(deps)

    parent_config = {"metadata": {"marker": "parent-turn"}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "how many orders?"}]},
        parent_config,
        context=TurnContext(user_id="exec", session_id="s1", turn_id="t1"),
    )

    assert captured_configs, "the analyst never called its nested agent.invoke"
    nested_config = captured_configs[0]
    assert nested_config is not None, (
        "the nested invoke was called with config=None instead of the "
        "parent's RunnableConfig"
    )
    assert nested_config.get("metadata", {}).get("marker") == "parent-turn", (
        "the nested invoke did not receive the parent run's config — got "
        f"{nested_config!r}"
    )


def test_three_queries_accumulate_three_attempts(make_deps):
    """The reducer's whole job. Without `operator.add` the second `run_sql`
    replaces the first and the turn reports one query where it ran three.

    This drives `build_analyst_tools` directly through a `create_agent`
    compiled with `state_schema=TurnState` — the SQL loop itself, which is
    what this task owns — rather than through the supervisor's `analyst`
    tool. `analyst` (in `subagents.py`) still returns a plain string today;
    lifting the subagent's `TurnState` keys into the parent turn's state is
    explicitly Task 3's job ("what Task 3's `analyst` tool lifts into the
    parent turn"), and `subagents.py` is off limits here. Asserting through
    the supervisor's top-level `agent.invoke(...)` result, as an earlier
    draft of this test did, produces `result["attempts"] == []` — not because
    the reducer or `index=len(runtime.state...)` logic is wrong (both are
    exercised and proven correct below), but because nothing yet copies the
    inner subagent's state into the outer graph. That gap is real and is
    left for Task 3 to close.
    """
    import pandas as pd

    from langchain.agents import create_agent

    from retail_agent.agent.state import TurnState
    from retail_agent.agent.tools import build_analyst_tools
    from .conftest import FakeSource

    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})})
    deps = make_deps(
        script=[
            [("run_sql", {"sql": "SELECT 1"})],
            [("run_sql", {"sql": "SELECT 2"})],
            [("run_sql", {"sql": "SELECT 3"})],
            "Three of them.",
        ],
        src=source,
    )
    agent = create_agent(
        model=deps.llm,
        tools=build_analyst_tools(deps),
        state_schema=TurnState,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "how many?"}]})

    assert [a["step_id"] for a in result["attempts"]] == ["q1", "q2", "q3"]
    assert [a["sql"] for a in result["attempts"]] == ["SELECT 1", "SELECT 2", "SELECT 3"]


def test_two_parallel_run_sql_calls_in_one_turn_do_not_crash_it(make_deps):
    """Reproduces the C2 regression: with no reducer on `frame` and
    `executed_sql`, LangGraph raises `InvalidUpdateError` — "can receive only
    one value per step" — the instant two tool calls in the same super-step
    both write one. `ScriptedChatModel`'s script entries are ordinarily one
    tool call per assistant turn; this one is two, one round, the shape
    Gemini and OpenAI both emit routinely and the shape the suite never had.
    Green here is the whole fix for `frame`/`executed_sql`'s `_keep_last`
    reducer.
    """
    import pandas as pd

    from langchain.agents import create_agent

    from retail_agent.agent.state import TurnState
    from retail_agent.agent.tools import build_analyst_tools
    from .conftest import FakeSource

    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})})
    deps = make_deps(
        script=[
            [
                ("run_sql", {"sql": "SELECT 1"}),
                ("run_sql", {"sql": "SELECT 2"}),
            ],
            "Two of them.",
        ],
        src=source,
    )
    agent = create_agent(
        model=deps.llm,
        tools=build_analyst_tools(deps),
        state_schema=TurnState,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "how many?"}]})

    assert len(result["attempts"]) == 2
    assert result["calls"] == 2
    assert result["frame"] is not None, "one of the two parallel writes survived"
    assert result["executed_sql"].startswith(("SELECT 1", "SELECT 2"))
    # Known, documented gap rather than a silent one: every tool call in a
    # parallel batch reads `runtime.state["attempts"]` before the super-step
    # that contains all of them has applied anything, so both compute the
    # same `index` and both attempts are numbered `q1` — unlike the
    # sequential `q1, q2, q3` in `test_three_queries_accumulate_three_attempts`
    # above. This does not corrupt `/metrics` (`compute_metrics` only reads
    # `attempts[0]`), but `/trace` would print two rows both saying `q1`.
    # There is no per-super-step running counter for a reducer to read, so
    # fixing this would need a different numbering scheme entirely — out of
    # scope for the crash this test guards against.
    assert [a["step_id"] for a in result["attempts"]] == ["q1", "q1"]


def test_a_failed_query_is_repaired_through_the_wired_stack(make_deps):
    """Proves `_SqlFailureRecorder` is actually installed, not just that the
    class works — `test_tools.py`'s `test_a_failed_query_is_recorded_and_repaired`
    already proves the class itself by calling `wrap_tool_call` directly, and
    that test would stay green even if `analyst_middleware` never listed
    `_SqlFailureRecorder()` at all.

    `_SqlFailureRecorder` replaced `ToolErrorMiddleware(on_error=describe_
    failure)` in `analyst_middleware`'s stack rather than joining it, so if
    that line were ever deleted there would be *no* handler left for
    `QuerySyntaxError` in the analyst loop: the raise would propagate straight
    out of `agent.invoke`, and the whole turn would die on a failure the loop
    is supposed to repair. Only a test that drives a real failing query
    through the real, fully assembled `analyst_middleware(deps)` stack can
    catch that — this is that test.
    """
    import pandas as pd

    from langchain.agents import create_agent

    from retail_agent.agent.middleware import analyst_middleware
    from retail_agent.agent.state import TurnState
    from retail_agent.agent.tools import build_analyst_tools
    from .conftest import FakeSource

    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})}, failing={"bad"})
    deps = make_deps(
        script=[
            [("run_sql", {"sql": "SELECT bad FROM users"})],
            [("run_sql", {"sql": "SELECT n FROM users"})],
            "One.",
        ],
        src=source,
    )
    agent = create_agent(
        model=deps.llm,
        tools=build_analyst_tools(deps),
        state_schema=TurnState,
        middleware=analyst_middleware(deps),
    )

    # If _SqlFailureRecorder is missing from the stack, this raises
    # QuerySyntaxError instead of returning — the turn dies instead of
    # completing with a repaired second attempt.
    result = agent.invoke({"messages": [{"role": "user", "content": "how many?"}]})

    assert [a["step_id"] for a in result["attempts"]] == ["q1", "q2"]
    assert result["attempts"][0]["error"], "the failed first attempt is recorded"
    assert result["attempts"][1]["row_count"] == 1, "the repaired second attempt ran"
    assert result["calls"] == 2
    assert result["messages"][-1].content == "One."


def test_two_analyst_calls_in_one_turn_do_not_collide_on_step_id(make_deps):
    """`step_id` used to always start at `q1` inside the analyst's own nested
    `agent.invoke` — that subgraph's state starts empty every time it is
    called, so it has no way to know a second `analyst` call is this turn's
    second query. `[a["step_id"] for a in result.update["attempts"]]` used to
    read `["q1"]` for both calls; renumbering against the parent turn's own
    running total (known only at the lift, in `subagents.py`) is what fixes
    it.
    """
    import pandas as pd

    from .conftest import FakeSource

    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})})
    deps = make_deps(
        script=[
            [("run_sql", {"sql": "SELECT 1"})],
            "one.",
            [("run_sql", {"sql": "SELECT 2"})],
            "two.",
        ],
        src=source,
    )
    analyst = subagents_for(deps)

    first = analyst["analyst"]("how many orders?", runtime=_runtime())
    second = analyst["analyst"](
        "how many customers?",
        runtime=_runtime(state={"attempts": first.update["attempts"]}),
    )

    assert [a["step_id"] for a in first.update["attempts"]] == ["q1"]
    assert [a["step_id"] for a in second.update["attempts"]] == ["q2"]


def test_the_analysts_redactions_reach_the_turn(make_deps):
    """Load-bearing lift, previously untested: without it, `render_answer`'s
    "N personal-data values masked" footnote, `TraceRecord.redactions` and
    `compute_metrics`'s masked total all read 0 for every `analyst` turn,
    because the nested subgraph's `redactions` never left its own state.
    """
    import pandas as pd

    from .conftest import FakeSource

    source = FakeSource(
        frames={"default": pd.DataFrame({"id": [1], "email": ["a@b.com"]})}
    )
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT id, email FROM users"})], "one."],
        src=source,
    )
    analyst = subagents_for(deps)

    result = analyst["analyst"]("who are our customers?", runtime=_runtime())

    assert result.update["redactions"] > 0


def test_the_analysts_calls_reach_the_turn(make_deps):
    """Load-bearing lift, previously untested: without `result.get("calls",
    0) + 1`, `AgentAnswer.calls` undercounts by one per `analyst` call — the
    nested `run_sql` call the subgraph made is invisible to the parent turn,
    and only the wrapper's own call is counted.
    """
    import pandas as pd

    from .conftest import FakeSource

    source = FakeSource(frames={"default": pd.DataFrame({"n": [1]})})
    deps = make_deps(
        script=[[("run_sql", {"sql": "SELECT COUNT(*) AS n FROM users"})], "one."],
        src=source,
    )
    analyst = subagents_for(deps)

    result = analyst["analyst"]("how many customers?", runtime=_runtime())

    # 1 for the nested run_sql call, +1 for the analyst call itself.
    assert result.update["calls"] == 2
