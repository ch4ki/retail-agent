from langchain_core.messages import AIMessage, HumanMessage

from retail_agent.agent.nodes.report_ops import report_ops_node
from retail_agent.agent.state import TurnState, fresh_scratch


def state_for(question, history=()):
    messages = []
    for role, text in history:
        messages.append(
            HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        )
    messages.append(HumanMessage(content=question))
    state = TurnState(messages=messages, user_id="dana", session_id="s1")
    state.update(fresh_scratch(repair_budget=3))
    return state


CONVERSATION = (
    ("user", "what was revenue in March 2024?"),
    ("assistant", "Revenue in March 2024 was $1,284,000 across 8,412 orders."),
)


def test_save_stores_a_report_and_confirms_with_its_id(make_deps, reports):
    deps = make_deps(
        [
            {"kind": "save", "title": "March 2024 revenue review"},
            "## Summary\n\nRevenue was $1,284,000.\n\n"
            "## Action items\n1. Investigate the April dip.",
        ],
        store=reports,
    )

    result = report_ops_node(
        state_for("save that as a report with action items", CONVERSATION), deps
    )

    saved = reports.list_reports(owner_id="dana")
    assert len(saved) == 1
    assert saved[0].title == "March 2024 revenue review"
    assert "Action items" in saved[0].body
    assert saved[0].id in result["answer"]


def test_saved_report_is_scanned_for_pii_before_storage(make_deps, reports):
    deps = make_deps(
        [
            {"kind": "save", "title": "Top customers"},
            "Our best customer is ada@example.com with $900 of spend.",
        ],
        store=reports,
    )

    report_ops_node(state_for("save that", CONVERSATION), deps)

    body = reports.list_reports(owner_id="dana")[0].body
    assert "ada@example.com" not in body
    assert "[redacted:email]" in body


def test_save_uses_the_conversation_as_its_source(make_deps, reports):
    """Frames are per-turn scratch and are already blank by the time a user says
    "save that", so the body is synthesised from history."""
    deps = make_deps(
        [{"kind": "save", "title": "March"}, "Body referencing the numbers."],
        store=reports,
    )

    report_ops_node(state_for("save that as a report", CONVERSATION), deps)

    assert "$1,284,000" in deps.llm.prompts[-1]


def test_list_returns_titles_without_a_second_model_call(make_deps, reports):
    reports.save(owner_id="dana", session_id="s1", title="Q1 review", body="x")
    deps = make_deps([{"kind": "list"}], store=reports)

    result = report_ops_node(state_for("show me my reports"), deps)

    assert "Q1 review" in result["answer"]
    assert len(deps.llm.prompts) == 1


def test_an_unreadable_reply_falls_back_to_listing(make_deps, reports):
    """The safe default. Guessing "delete" from a reply we could not read is
    how you delete someone's work."""
    deps = make_deps(["not structured output at all"], store=reports)

    result = report_ops_node(state_for("do something to my reports"), deps)

    assert "no saved reports" in result["answer"].lower()


def test_irrelevant_fields_are_discarded_for_non_deletes(make_deps, reports):
    """Constrained decoding makes the model populate every field — a live spike
    returned session_scoped=True for "what have I saved?". Those fields decide
    what a delete targets, so they are only trusted on a delete."""
    from retail_agent.agent.nodes.report_ops import extract_op

    deps = make_deps(
        [{"kind": "save", "title": "T", "term": "junk", "session_scoped": True}],
        store=reports,
    )

    op = extract_op(state_for("save that"), deps)

    assert op.kind == "save"
    assert op.term == ""
    assert op.session_scoped is False


from langgraph.checkpoint.memory import MemorySaver

from retail_agent.agent.graph import build_graph


def _seed(reports, n, term="Calvin Klein"):
    return [
        reports.save(
            owner_id="dana",
            session_id="s1",
            title=f"Report {i}",
            body=f"{term} performed well",
        )
        for i in range(n)
    ]


def _start(deps, question="delete reports mentioning Calvin Klein"):
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s1"}}
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_id": "dana",
            "session_id": "s1",
        },
        config,
    )
    return graph, config, state


DELETE_CK = [{"intent": "report_op"}, {"kind": "delete", "term": "Calvin Klein"}]


def test_delete_pauses_before_touching_anything(make_deps, reports):
    _seed(reports, 3)
    deps = make_deps(DELETE_CK, store=reports)

    graph, config, _ = _start(deps)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("await_confirmation",)
    pending = snapshot.values["pending_action"]
    assert pending.token == "DELETE 3"
    assert set(pending.titles) == {"Report 0", "Report 1", "Report 2"}
    assert len(reports.list_reports(owner_id="dana")) == 3, "nothing deleted yet"


def test_typed_confirmation_deletes_and_audits(make_deps, reports):
    _seed(reports, 3)
    deps = make_deps(DELETE_CK, store=reports)
    graph, config, _ = _start(deps)

    graph.update_state(
        config, {"confirmation": "DELETE 3"}, as_node="await_confirmation"
    )
    final = graph.invoke(None, config)

    assert reports.list_reports(owner_id="dana") == []
    assert reports.last_action(owner_id="dana").token == "DELETE 3"
    assert "3" in final["answer"]


def test_wrong_confirmation_changes_nothing(make_deps, reports):
    _seed(reports, 3)
    deps = make_deps(DELETE_CK, store=reports)
    graph, config, _ = _start(deps)

    graph.update_state(
        config, {"confirmation": "yes please"}, as_node="await_confirmation"
    )
    final = graph.invoke(None, config)

    assert len(reports.list_reports(owner_id="dana")) == 3
    assert "nothing" in final["answer"].lower()


def test_a_single_report_uses_an_inline_confirmation(make_deps, reports):
    _seed(reports, 1)
    deps = make_deps(DELETE_CK, store=reports)

    graph, config, _ = _start(deps, "delete the Calvin Klein report")

    assert graph.get_state(config).values["pending_action"].token == "y"


def test_matching_nothing_never_pauses(make_deps, reports):
    deps = make_deps(
        [{"intent": "report_op"}, {"kind": "delete", "term": "Nothing"}], store=reports
    )

    graph, config, state = _start(deps, "delete reports mentioning Nothing")

    assert graph.get_state(config).next == ()
    assert "no reports" in state["answer"].lower()


def test_session_scope_resolves_by_session_id(make_deps, reports):
    reports.save(owner_id="dana", session_id="s1", title="This one", body="x")
    reports.save(owner_id="dana", session_id="other", title="Not this", body="x")
    deps = make_deps(
        [{"intent": "report_op"}, {"kind": "delete", "session_scoped": True}],
        store=reports,
    )

    graph, config, _ = _start(deps, "delete the reports from this conversation")

    # A checkpoint round-trip turns tuples into lists, so compare as a sequence.
    staged = graph.get_state(config).values["pending_action"]
    assert list(staged.titles) == ["This one"]


def test_resolving_node_does_not_re_execute_on_resume(make_deps, reports):
    """The reason the gate is a breakpoint rather than an in-node interrupt.
    Two replies are queued and two are consumed; a re-executing node would ask
    for a third and the scripted LLM would raise."""
    _seed(reports, 2)
    deps = make_deps(DELETE_CK, store=reports)
    graph, config, _ = _start(deps)

    graph.update_state(
        config, {"confirmation": "DELETE 2"}, as_node="await_confirmation"
    )
    graph.invoke(None, config)

    assert deps.llm.replies == []
