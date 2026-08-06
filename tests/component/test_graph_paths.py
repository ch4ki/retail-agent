import pandas as pd

from retail_agent.agent.graph import build_graph, run_turn
from tests.component.conftest import FakeSource


def turn(graph, question="top customers by spend"):
    return run_turn(
        graph, user_id="dana", session_id="s1", question=question
    )


def test_schema_question_never_executes_sql(make_deps, source):
    deps = make_deps([{"intent": "schema"}, "We hold orders, products and customers."], src=source)
    state = turn(build_graph(deps), "what data do you have?")

    assert source.executed == []
    assert "orders" in state["answer"]


def test_chat_follow_up_never_executes_sql(make_deps, source):
    deps = make_deps([{"intent": "chat"}, "Glad that helped."], src=source)
    state = turn(build_graph(deps), "thanks, that's useful")

    assert source.executed == []
    assert state["answer"] == "Glad that helped."


def test_happy_path_produces_an_answer(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["total spend per customer"]},
            "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",
            "Your top customer spent $100.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert "$100" in state["answer"]
    assert len(source.executed) == 1


def test_a_first_try_query_records_exactly_one_attempt(make_deps, source):
    """One row per attempt, not per node visit. `len(sql_attempts)` is what the
    CLI shows the user as "N query attempts" and what phase 2 divides by for the
    self-correction rate, so a clean first try must count as one."""
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["total spend per customer"]},
            "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",
            "Your top customer spent $100.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert len(state["sql_attempts"]) == 1
    attempt = state["sql_attempts"][0]
    assert attempt.row_count == 2
    assert "`bigquery-public-data" not in attempt.sql, "sql is what the model wrote"
    assert "`bigquery-public-data" in attempt.executed_sql, "and this is what ran"


def test_two_failures_still_leave_room_for_a_third_attempt(make_deps, source):
    """Runs on the configured default rather than the test fixture's budget, so
    it fails if the default stops matching what the README promises."""
    from retail_agent.agent.deps import AgentDeps
    from retail_agent.config import Settings
    from retail_agent.safety.pii import PiiPolicy
    from retail_agent.obs.traces import InMemoryTraceStore
    from retail_agent.store.memory_reports import InMemoryReportStore
    from tests.component.conftest import ScriptedLLM

    broken = FakeSource(
        frames={"ok": pd.DataFrame({"id": [1], "spend": [10]})}, failing={"BROKEN"}
    )
    deps = AgentDeps(
        settings=Settings(_env_file=None, google_cloud_project="test"),
        llm=ScriptedLLM(
            [
                {"intent": "analyze"},
                {"steps": ["revenue"]},
                "SELECT BROKEN FROM users",  # fails in the warehouse
                "SELECT email AS c FROM users",  # rejected by the guard
                "SELECT id, spend FROM users",  # third time lucky
                "Revenue was $10.",
            ]
        ),
        source=broken,
        policy=PiiPolicy.default(),
        reports=InMemoryReportStore(),
        traces=InMemoryTraceStore(),
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert len(state["sql_attempts"]) == 3


def test_guard_violation_triggers_one_repair_then_succeeds(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["top customers"]},
            "SELECT email AS contact FROM users",  # rejected by the guard
            "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",  # repaired
            "Your top customer spent $100.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert len(state["sql_attempts"]) >= 2
    assert source.executed, "the repaired query should have run"


def test_exhausted_repair_budget_degrades_instead_of_looping(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["top customers"]},
            "SELECT email AS contact FROM users",
            "SELECT email AS contact FROM users",
            "SELECT email AS contact FROM users",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "degraded"
    assert source.executed == []
    assert "couldn't build a working query" in state["answer"]


def test_syntax_error_is_repaired(make_deps):
    broken = FakeSource(
        frames={"ok": pd.DataFrame({"id": [1], "spend": [10]})}, failing={"BROKEN"}
    )
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue"]},
            "SELECT BROKEN FROM users",
            "SELECT id, spend FROM users",
            "Revenue was $10.",
        ],
        src=broken,
    )
    state = turn(build_graph(deps))

    assert state["status"] == "ok"
    assert len(broken.executed) == 2


def test_multi_step_plan_runs_every_step(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue for brand X", "revenue for brand Y"]},
            "SELECT id, spend FROM users",
            "SELECT id, spend FROM users",
            "Brand X leads brand Y.",
        ],
        src=source,
    )
    state = turn(build_graph(deps), "compare brand X and brand Y")

    assert len(source.executed) == 2
    assert len(state["frames"]) == 2


# The warehouse returns an `email` column regardless of the SQL, so these two
# exercise masking at the data boundary rather than the guard's projection rule.


def test_pii_never_reaches_the_model_context(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["top customers"]},
            "SELECT id, spend FROM users",
            "Top customer identified.",
        ],
        src=source,
    )
    turn(build_graph(deps))

    synthesis_prompts = [p for p in deps.llm.prompts if "Query results" in p]
    assert synthesis_prompts, "synthesis should have been reached"
    assert "a@b.com" not in synthesis_prompts[-1]
    assert "@" not in synthesis_prompts[-1].split("Query results")[1]


def test_masked_frames_are_stored_not_raw_ones(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["top customers"]},
            "SELECT id, spend FROM users",
            "Done.",
        ],
        src=source,
    )
    state = turn(build_graph(deps))

    stored = state["frames"]["step_1"]
    assert "@" not in str(stored.column("email")[0])
    assert state["redactions"] == 2


def test_guard_blocks_unmaskable_pii_projections_end_to_end(make_deps, source):
    # A live model produced the CONCAT(MAX(...)) form to work around the guard.
    # It renames the output column, so masking cannot find it — the query must
    # never reach the warehouse.
    evasion = (
        "SELECT u.id AS user_id, "
        "CONCAT(MAX(u.first_name), ' ', MAX(u.last_name)) AS user_name "
        "FROM users AS u GROUP BY user_id"
    )
    deps = make_deps(
        [{"intent": "analyze"}, {"steps": ["customer names"]}, evasion, evasion, evasion], src=source
    )
    state = turn(build_graph(deps), "list our top customers by name")

    assert source.executed == [], "an unmaskable PII query must never run"
    assert state["status"] == "degraded"


def test_bare_pii_column_runs_and_is_masked_end_to_end(make_deps, source):
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["customer contacts"]},
            "SELECT id, email FROM users",
            "Listed 2 customers.",
        ],
        src=source,
    )
    state = turn(build_graph(deps), "show me customer records")

    assert len(source.executed) == 1, "a maskable query is allowed to run"
    assert state["redactions"] == 2
    assert "@" not in str(state["frames"]["step_1"].column("email")[0])


def test_state_survives_checkpointing(make_deps, source):
    """The CLI always runs with a checkpointer, so every value in TurnState
    must be serialisable. Running without one hid a crash on the first
    analysis turn."""
    from langgraph.checkpoint.memory import MemorySaver

    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["total spend per customer"]},
            "SELECT id, spend FROM users",
            "Your top customer spent $100.",
        ],
        src=source,
    )
    graph = build_graph(deps, checkpointer=MemorySaver())
    state = run_turn(
        graph,
        user_id="dana",
        session_id="s1",
        question="top customers",
        config={"configurable": {"thread_id": "s1"}},
    )

    assert state["status"] == "ok"
    assert state["frames"]["step_1"].row_count == 2


def test_an_empty_plan_terminates_instead_of_looping(make_deps, source):
    """draft_sql returns nothing when there is no current step and spends no
    budget, so routing back to it would cycle until the recursion limit."""
    from retail_agent.agent.graph import _after_draft
    from retail_agent.agent.state import TurnState

    assert _after_draft(TurnState(plan=[], step_index=0, repair_budget=2)) == "synthesize"


def test_blank_turn_runs_end_to_end_without_touching_the_warehouse(make_deps, source):
    from langchain_core.messages import HumanMessage

    deps = make_deps([])  # no replies queued: any model call fails the test
    graph = build_graph(deps)
    state = graph.invoke({"messages": [HumanMessage(content="   ")]})

    assert source.executed == []
    assert deps.llm.prompts == []
    assert "?" in state["answer"]


# Studio invokes the compiled graph with nothing but `messages`, so none of the
# keys new_turn_state seeds are present. These two cover that entrypoint; every
# test above goes through run_turn and so cannot see a missing budget.


def test_studio_shaped_invoke_still_gets_a_repair_budget(make_deps):
    from langchain_core.messages import HumanMessage

    broken = FakeSource(
        frames={"ok": pd.DataFrame({"id": [1], "spend": [10]})}, failing={"BROKEN"}
    )
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["revenue"]},
            "SELECT BROKEN FROM users",  # fails in the warehouse
            "SELECT id, spend FROM users",  # repaired
            "Revenue was $10.",
        ],
        src=broken,
    )
    graph = build_graph(deps)
    state = graph.invoke({"messages": [HumanMessage(content="revenue so far")]})

    assert state["status"] == "ok"
    assert len(broken.executed) == 2, "the failure should have been repaired"


def test_second_analysis_turn_starts_with_a_fresh_budget(make_deps):
    """Studio keeps thread state between turns. A budget spent on turn one must
    not carry over, or every later turn degrades on its first failure."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    broken = FakeSource(
        frames={"ok": pd.DataFrame({"id": [1], "spend": [10]})}, failing={"BROKEN"}
    )
    deps = make_deps(
        [
            # turn one: fails twice, exhausting the budget, then degrades
            # (the degraded answer is templated, so no synthesis reply is used)
            {"intent": "analyze"},
            {"steps": ["revenue"]},
            "SELECT BROKEN FROM users",
            "SELECT BROKEN FROM users",
            # turn two: one failure, then a repair that works
            {"intent": "analyze"},
            {"steps": ["revenue again"]},
            "SELECT BROKEN FROM users",
            "SELECT id, spend FROM users",
            "Revenue was $10.",
        ],
        src=broken,
    )
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    first = graph.invoke({"messages": [HumanMessage(content="revenue?")]}, config)
    assert first["status"] == "degraded"

    second = graph.invoke({"messages": [HumanMessage(content="and now?")]}, config)
    assert second["status"] == "ok"


def test_a_failed_turn_does_not_answer_from_the_previous_turns_results(make_deps):
    """Step ids restart at step_1 every turn. If frames carry over, turn two's
    failed step_1 is masked by turn one's succeeded step_1: synthesize sees a
    full result set, reports no missing steps, and narrates last turn's numbers
    as the answer to a new question — confidently, and marked ok."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    broken = FakeSource(
        frames={"ok": pd.DataFrame({"id": [1], "spend": [10]})}, failing={"BROKEN"}
    )
    deps = make_deps(
        [
            # turn one succeeds and leaves a frame behind
            {"intent": "analyze"},
            {"steps": ["revenue"]},
            "SELECT id, spend FROM users",
            "Revenue was $10.",
            # turn two fails every attempt, so it has nothing of its own to say
            {"intent": "analyze"},
            {"steps": ["revenue for March"]},
            "SELECT BROKEN FROM users",
            "SELECT BROKEN FROM users",
            "Revenue was $10.",  # only reachable if stale frames leak through
        ],
        src=broken,
    )
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    first = graph.invoke({"messages": [HumanMessage(content="revenue?")]}, config)
    assert first["status"] == "ok", "turn one should succeed"

    second = graph.invoke({"messages": [HumanMessage(content="and March?")]}, config)

    assert second["frames"] == {}
    assert second["status"] == "degraded"


# Observability: the graph records what it did, so a turn can be explained
# after the fact rather than guessed at from the answer text.


def test_every_node_visit_records_a_timed_event(make_deps, source):
    deps = make_deps([{"intent": "schema"}, "We hold orders and customers."], src=source)

    state = turn(build_graph(deps), "what data do you have?")

    assert [e.node for e in state["events"]] == ["start_turn", "route", "schema"]
    assert all(e.duration_ms >= 0 for e in state["events"])


def test_events_record_the_repair_loop(make_deps, source):
    """The count in the footnote says "3 query attempts" and nothing else. The
    event log is what distinguishes three steps from one step failing twice."""
    deps = make_deps(
        [
            {"intent": "analyze"},
            {"steps": ["top customers"]},
            "SELECT email AS contact FROM users",  # guard rejects
            "SELECT id, SUM(spend) AS spend FROM users GROUP BY id",
            "Your top customer spent $100.",
        ],
        src=source,
    )

    state = turn(build_graph(deps))

    assert [e.node for e in state["events"]] == [
        "start_turn",
        "route",
        "recall",
        "plan",
        "draft_sql",
        "draft_sql",
        "execute",
        "synthesize",
    ]


def test_events_do_not_leak_into_the_next_turn(make_deps, source):
    from langgraph.checkpoint.memory import MemorySaver

    deps = make_deps(
        [
            {"intent": "chat"},
            "Glad that helped.",
            {"intent": "chat"},
            "Any time.",
        ],
        src=source,
    )
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s1"}}

    run_turn(graph, user_id="d", session_id="s1", question="thanks", config=config)
    second = run_turn(
        graph, user_id="d", session_id="s1", question="thanks again", config=config
    )

    assert [e.node for e in second["events"]] == ["start_turn", "route", "chat"]
