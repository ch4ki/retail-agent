"""The report library, and the gate in front of the destructive part.

The confirmation is an interrupt `delete_reports` raises from inside itself,
after resolving what it would delete but before writing anything, so the
assertions here are about what is still in the store while the user is being
asked — not about what the agent said.
"""

from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from retail_agent.agent.deps import TurnContext
from retail_agent.agent.middleware import MissingTurnIdentity
from retail_agent.agent.reports import build_report_tools, confirmation_token
from retail_agent.agent.subagents import build_subagents
from retail_agent.agent.supervisor import build_agent


def _runtime_with(context):
    """A ToolRuntime carrying just the context, for calling a tool's raw
    `.func` directly the way this file's other tests do.

    Six of its nine fields are required; `tools`, `execution_info` and
    `server_info` have defaults and are omitted.
    """
    from langchain.tools import ToolRuntime

    return ToolRuntime(
        state=None,
        context=context,
        config={},
        stream_writer=None,
        tool_call_id="test",
        store=None,
    )


def context_for(user_id: str = "exec", session_id: str = "s1", turn_id: str = "t1") -> TurnContext:
    """The runtime context for a test's identity.

    Used everywhere a test drives a compiled agent through `agent.invoke`
    directly: the tools read identity from `runtime.context`, which `invoke`
    never fills in unless the caller passes it, so a test that omits it would
    have every identity-scoped tool see an empty user rather than the one the
    fixtures set up.
    """
    return TurnContext(user_id=user_id, session_id=session_id, turn_id=turn_id)


def _text(result):
    """A tool's answer, unwrapped from the `Command` it now returns — the one
    `ToolMessage` on `update["messages"]` carries the text a model would see."""
    if isinstance(result, Command):
        return result.update["messages"][0].content
    return result


def writer(deps):
    func = {t.name: t.func for t in build_subagents(deps)}["report_writer"]

    def call(**kwargs):
        return func(runtime=_runtime_with(context_for()), **kwargs)

    return call


@pytest.fixture
def saved(reports):
    reports.save(
        owner_id="exec", session_id="s1", title="Acme Q1", body="Acme did well."
    )
    reports.save(
        owner_id="exec", session_id="s1", title="Beta Q1", body="Beta did badly."
    )
    return reports


def run(deps, question, saver=None):
    saver = saver or MemorySaver()
    agent = build_agent(deps, checkpointer=saver)
    config = {"configurable": {"thread_id": "s1"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
        context=context_for(),
    )
    return agent, config, result


def test_a_delete_matching_nothing_never_raises_a_prompt(make_deps, saved):
    """Confirming a no-op is how a confirmation prompt becomes something people
    click through without reading."""
    deps = make_deps(
        script=[[("delete_reports", {"term": "Nothing"})], "I found none."]
    )
    _, _, result = run(deps, "delete the reports about Nothing")

    assert not result.get("__interrupt__")
    assert len(saved.list_reports(owner_id="exec")) == 2


def test_nothing_is_deleted_while_the_user_is_being_asked(make_deps, saved):
    """The write cannot have happened: the tool has not run yet."""
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted."]
    )
    _, _, result = run(deps, "delete all reports mentioning Acme")

    assert result.get("__interrupt__")
    assert len(saved.list_reports(owner_id="exec")) == 2
    payload = result["__interrupt__"][0].value
    assert "Acme Q1" in payload["manifest"]
    assert "Beta Q1" not in payload["manifest"], "the other report must not be caught up in the match"


def test_approving_deletes_exactly_what_was_shown(make_deps, saved):
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted 1 report."]
    )
    agent, config, result = run(deps, "delete all reports mentioning Acme")
    payload = result["__interrupt__"][0].value

    agent.invoke(
        Command(
            resume={
                "approved": True,
                "report_ids": payload["report_ids"],
                "token": payload["token"],
            }
        ),
        config,
        context=context_for(),
    )

    remaining = [r.title for r in saved.list_reports(owner_id="exec")]
    assert remaining == ["Beta Q1"]


def test_a_resumed_approval_with_no_identity_is_refused_not_silent(make_deps, saved):
    """The LangGraph server's exact exposure: a resume posted with no
    `context` in its body coerces to `TurnContext()` — every field `""`
    rather than `None` — via `_coerce_context`. Before the identity guard
    moved to `wrap_tool_call`, that shape sailed straight through: the
    replayed `delete_reports` body called `resolve_delete` for owner `""`,
    found nothing, returned early, and the approval the executive just typed
    by hand was discarded with no error and nothing said. `before_agent`
    cannot catch this — it is a completed checkpoint node, so
    `Command(resume=...)` never re-runs it; only a hook that lives inside
    the replayed tools node itself can.
    """
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted 1 report."]
    )
    agent, config, result = run(deps, "delete all reports mentioning Acme")
    payload = result["__interrupt__"][0].value

    with pytest.raises(MissingTurnIdentity):
        agent.invoke(
            Command(
                resume={
                    "approved": True,
                    "report_ids": payload["report_ids"],
                    "token": payload["token"],
                }
            ),
            config,
            context=TurnContext(),
        )

    assert len(saved.list_reports(owner_id="exec")) == 2, "nothing was deleted"


def test_a_paused_delete_traces_exactly_one_event(make_deps, saved):
    """`interrupt()` raises before the tool returns a `Command` on the
    pre-pause pass, so nothing is recorded then; `interrupt()` replays the
    whole node on resume, and that pass is the only one that ever returns an
    `events` entry — there must be exactly one, and it must carry the real
    detail.
    """
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted 1 report."]
    )
    agent, config, result = run(deps, "delete all reports mentioning Acme")
    payload = result["__interrupt__"][0].value

    final = agent.invoke(
        Command(
            resume={
                "approved": True,
                "report_ids": payload["report_ids"],
                "token": payload["token"],
            }
        ),
        config,
        context=context_for(),
    )

    delete_events = [e for e in final["events"] if e["name"] == "delete_reports"]
    assert len(delete_events) == 1, final["events"]
    assert delete_events[0]["detail"] == "deleted 1"
    assert final["calls"] == 1


def test_rejecting_leaves_the_library_alone(make_deps, saved):
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Nothing was deleted."]
    )
    agent, config, _ = run(deps, "delete all reports mentioning Acme")

    agent.invoke(
        Command(resume={"approved": False}), config, context=context_for()
    )

    assert len(saved.list_reports(owner_id="exec")) == 2


def test_a_deletion_can_be_undone(make_deps, saved):
    """Soft delete plus an audit row is what makes /undo possible at all."""
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted."]
    )
    agent, config, result = run(deps, "delete all reports mentioning Acme")
    payload = result["__interrupt__"][0].value
    agent.invoke(
        Command(
            resume={
                "approved": True,
                "report_ids": payload["report_ids"],
                "token": payload["token"],
            }
        ),
        config,
        context=context_for(),
    )

    assert saved.undo(owner_id="exec") == 1
    assert len(saved.list_reports(owner_id="exec")) == 2


def test_the_manifest_names_every_report_and_the_token_scales(make_deps, saved):
    """One report is a low-stakes correction; several is not."""
    deps = make_deps(script=[[("delete_reports", {"term": ""})], "Deleted."])
    _, _, result = run(deps, "delete all my reports")

    payload = result["__interrupt__"][0].value
    assert set(payload["report_ids"]) == {r.id for r in saved.list_reports(owner_id="exec")}
    assert payload["token"] == confirmation_token(2) == "DELETE 2"
    assert "Acme Q1" in payload["manifest"] and "Beta Q1" in payload["manifest"]


def test_a_delete_cannot_reach_another_owner(make_deps, saved):
    """Ownership is a WHERE clause in the store, not a check in the agent."""
    deps = make_deps(script=[[("delete_reports", {"term": "Acme"})], "None."])
    tools = {t.name: t.func for t in build_report_tools(deps)}

    answer = _text(tools["delete_reports"](
        runtime=_runtime_with(context_for(user_id="someone-else")), term="Acme"
    ))

    assert "no reports" in answer.lower()
    assert len(saved.list_reports(owner_id="exec")) == 2


def test_the_model_never_receives_the_report_it_wrote(make_deps):
    """The whole point: a body in the tool return is a body re-sent on every
    later model call, and a body the model can retype differently."""
    body = (
        "## Summary\nQ1 denim review.\n\n"
        "## What the data shows\nDenim fell 11.8% in Q1.\n\n"
        "## Action items\n1. Audit Texas."
    )
    deps = make_deps(script=[body])

    receipt = _text(writer(deps)(brief="denim findings", title="Q1 Denim"))

    assert "Denim fell 11.8%" not in receipt
    assert "Audit Texas" not in receipt
    assert "Q1 Denim" in receipt


def test_what_is_stored_is_what_the_writer_produced(make_deps, reports):
    """No model sits between the two any more, so this is now an identity
    rather than a hope.

    `TurnState["reports_written"]` never carries a body (see
    `test_a_reports_body_is_not_in_state` in `test_supervisor.py`) — the
    store's copy is the only one that exists, so there is nothing left to
    compare it against here beyond checking the store got what was written.
    """
    body = "## Summary\nDenim fell in Q1."
    deps = make_deps(script=[body])

    writer(deps)(brief="denim findings", title="Q1 Denim")

    stored = reports.list_reports(owner_id="exec")[0]
    assert stored.body == body


def test_a_report_is_scanned_when_it_is_written_not_when_it_is_saved(
    make_deps, reports
):
    """A report shown but never saved used to reach the executive unscanned,
    because the only scan lived inside `save_report`."""
    deps = make_deps(script=["Reach Dana at dana@example.com about this."])

    writer(deps)(brief="contacts", title="Contacts")

    assert "[redacted:email]" in reports.list_reports(owner_id="exec")[0].body


def test_the_show_flag_is_carried_to_the_cli(make_deps):
    """The model decides whether the executive asked to read this; the CLI only
    obeys."""
    deps = make_deps(script=["## Summary\nA draft."])

    result = writer(deps)(brief="b", title="T", show_to_executive=False)

    assert result.update["reports_written"][0]["show"] is False


def test_a_short_headerless_report_still_leaks_nothing(make_deps):
    """The adversarial shape. An excerpt-based receipt returned this whole body
    verbatim — a report short enough to fit in the receipt is still a report."""
    deps = make_deps(script=["Denim fell 11.8% in Q1. Texas drove it."])

    receipt = _text(writer(deps)(brief="b", title="T"))

    assert "Denim fell" not in receipt
    assert "Texas" not in receipt


def test_there_is_no_save_report_tool(make_deps):
    """Its only caller was the supervisor retyping a body it had just read."""
    from retail_agent.agent.supervisor import build_tools

    names = {t.name for t in build_tools(make_deps())}

    assert "save_report" not in names
    assert "report_writer" in names


def asker(deps):
    func = {t.name: t.func for t in build_subagents(deps)}["ask_about_report"]

    def call(**kwargs):
        return func(runtime=_runtime_with(context_for()), **kwargs)

    return call


def test_a_report_is_answered_from_its_stored_body(make_deps, reports):
    """The supervisor has only an id. If this did not load the body, the
    executive could never ask anything about a report they had saved."""
    saved = reports.save(
        owner_id="exec",
        session_id="s1",
        title="Q1 Denim",
        body="## Action items\n1. Audit Texas inventory depth.",
    )
    deps = make_deps(script=["Audit Texas inventory depth."])

    answer = _text(asker(deps)(report_id=saved.id, question="what were the actions?"))

    assert "Audit Texas" in answer
    assert "Audit Texas inventory depth." in deps.llm.prompts[0]


def test_another_users_report_is_not_readable(make_deps, reports):
    """Ownership is a property of the store's query, not of the caller. A bare
    id must not be enough."""
    saved = reports.save(
        owner_id="someone-else", session_id="s9", title="Theirs", body="Secret."
    )
    deps = make_deps(script=[])

    answer = _text(asker(deps)(report_id=saved.id, question="what does it say?"))

    assert "Secret" not in answer
    assert "list_reports" in answer


def test_a_missing_report_costs_no_model_call(make_deps):
    """An empty script raises if the subagent is built, so this asserts the
    early return rather than the wording."""
    deps = make_deps(script=[])

    answer = _text(asker(deps)(report_id="nope", question="?"))

    assert "No report nope" in answer


class _FlakyOnce:
    """A model that fails its first call with a transient error, then delegates.

    Wrapping rather than subclassing `ScriptedChatModel`: the script must be
    consumed by the successful attempt, not by the failed one.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit")
        return self._inner.invoke(*args, **kwargs)


def test_a_transient_failure_while_writing_the_report_is_survived(
    make_deps, reports
):
    """The defect this replaces: `report_writer` ran with no middleware, so a
    429 killed the turn — after the SQL had already been paid for."""
    base = make_deps(script=["## Summary\nDenim fell in Q1."])
    flaky = _FlakyOnce(base.llm)
    deps = replace(base, llm=flaky)

    writer(deps)(brief="denim findings", title="Q1 Denim")

    assert flaky.calls == 2, "the first attempt failed and the second succeeded"
    assert reports.list_reports(owner_id="exec")[0].body == "## Summary\nDenim fell in Q1."


def test_a_report_added_between_prompt_and_approval_is_not_deleted(
    make_deps, saved, reports
):
    """`interrupt()` replays the tool body, so `resolve_delete` runs again
    after the executive has already seen the manifest. The ids come back in
    the resume value precisely so the replay cannot widen the blast radius."""
    deps = make_deps(script=[[("delete_reports", {"term": "Q1"})], "Deleted."])
    saver = MemorySaver()
    agent = build_agent(deps, checkpointer=saver)
    config = {"configurable": {"thread_id": "s1"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "delete the Q1 reports"}]},
        config,
        context=context_for(),
    )
    payload = result["__interrupt__"][0].value
    shown = list(payload["report_ids"])

    # A third matching report lands while the executive is reading the prompt.
    reports.save(
        owner_id="exec", session_id="s1", title="Gamma Q1", body="Gamma did fine."
    )

    agent.invoke(
        Command(resume={"approved": True, "report_ids": shown, "token": payload["token"]}),
        config,
        context=context_for(),
    )

    surviving = {r.title for r in reports.list_reports(owner_id="exec")}
    assert surviving == {"Gamma Q1"}, "only the reports shown were deleted"


def test_a_transient_failure_while_answering_about_a_report_is_survived(
    make_deps, reports, monkeypatch
):
    """`ask_about_report` drives a working `ScriptedChatModel` in every other
    test here, so a bare `deps.llm.invoke` would be indistinguishable from
    `resilient_call` — this is the one test that tells them apart.

    `resilient_call`'s `sleep` default is bound to `time.sleep` at import
    time, so patching `time.sleep` itself would not reach it — the default is
    mutated in place instead, the same dict `resilient_call(deps, call)`
    reads from on every call that does not pass `sleep=` explicitly.
    """
    import retail_agent.llm.resilience as resilience

    monkeypatch.setitem(
        resilience.resilient_call.__kwdefaults__, "sleep", lambda seconds: None
    )

    saved = reports.save(
        owner_id="exec",
        session_id="s1",
        title="Q1 Denim",
        body="## Action items\n1. Audit Texas inventory depth.",
    )
    base = make_deps(script=["Audit Texas inventory depth."])
    flaky = _FlakyOnce(base.llm)
    deps = replace(base, llm=flaky)

    answer = _text(
        asker(deps)(report_id=saved.id, question="what were the actions?")
    )

    assert flaky.calls == 2, "the first attempt failed and the second succeeded"
    assert "Audit Texas" in answer


def test_one_compiled_agent_serves_two_users(make_deps, reports):
    """The shape this change exists for. Identity used to be decided when the
    agent was built, so a single compiled graph could only ever act for one
    executive. Sub-project E made the server rebuild per request, which hid the
    symptom; this removes the reason.

    Built ONCE — `build_agent` is called a single time, exactly like a
    long-lived server process compiles it once — and invoked twice with
    different `TurnContext`s. Calling a tool's raw `.func` directly, as this
    test used to, bypasses `context_schema`, the whole middleware stack and
    the framework's runtime injection: it would keep passing with
    `context_schema=TurnContext` deleted from `build_agent`, with
    `_identity_guard` removed from the middleware stack, and with runtime
    injection broken outright, because none of those are on the path a bare
    function call takes. Driving the compiled graph through
    `agent.invoke(..., context=...)`, the way the CLI and the server both
    actually call it, is what exercises all three.
    """
    body = "## Summary\nDenim fell in Q1."
    deps = make_deps(
        script=[
            [("report_writer", {"brief": "denim findings", "title": "Ada's report"})],
            body,
            "Written up for Ada.",
            [("report_writer", {"brief": "denim findings", "title": "Bo's report"})],
            body,
            "Written up for Bo.",
        ]
    )
    agent = build_agent(deps, checkpointer=MemorySaver())

    agent.invoke(
        {"messages": [{"role": "user", "content": "write it up"}]},
        {"configurable": {"thread_id": "ada-thread"}},
        context=TurnContext(user_id="ada", session_id="s1", turn_id="t1"),
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "write it up"}]},
        {"configurable": {"thread_id": "bo-thread"}},
        context=TurnContext(user_id="bo", session_id="s2", turn_id="t2"),
    )

    assert [r.title for r in reports.list_reports(owner_id="ada")] == ["Ada's report"]
    assert [r.title for r in reports.list_reports(owner_id="bo")] == ["Bo's report"]


def test_the_analysts_record_reaches_the_turn(make_deps):
    """`run_sql` writes the analyst SUBGRAPH's state. A nested invoke returns
    its state rather than merging it, so nothing carries those fields upward on
    its own — the `analyst` tool has to lift them, and this is what says so."""
    import pandas as pd

    from .conftest import FakeSource

    source = FakeSource(frames={"default": pd.DataFrame({"n": [7]})})
    deps = make_deps(
        script=[
            [("analyst", {"question": "how many?"})],
            [("run_sql", {"sql": "SELECT 7"})],
            "Seven.",
            "Seven.",
        ],
        src=source,
    )
    agent = build_agent(deps, checkpointer=MemorySaver())

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "how many?"}]},
        {"configurable": {"thread_id": "s1"}},
        context=TurnContext(user_id="exec", session_id="s1", turn_id="t1"),
    )

    assert result["executed_sql"], "the analyst's SQL never reached the turn"
    assert result["frame"] is not None
    assert [a["sql"] for a in result["attempts"]] == ["SELECT 7"]


def test_a_reports_body_is_not_in_state(make_deps, reports):
    """Bodies are the only large payload here and state is checkpointed on
    every super-step. The library's copy is the one that gets read."""
    body = "## Summary\nDenim fell in Q1."
    deps = make_deps(
        script=[
            [("report_writer", {"brief": "denim", "title": "Q1 Denim"})],
            body,
            "Written.",
        ]
    )
    agent = build_agent(deps, checkpointer=MemorySaver())

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "write it up"}]},
        {"configurable": {"thread_id": "s1"}},
        context=TurnContext(user_id="exec", session_id="s1", turn_id="t1"),
    )

    written = result["reports_written"][0]
    assert set(written) == {"report_id", "title", "show"}
    assert body not in str(result["reports_written"])
    assert reports.get(owner_id="exec", report_id=written["report_id"]).body == body
