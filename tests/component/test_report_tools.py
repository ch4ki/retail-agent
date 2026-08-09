"""The report library, and the gate in front of the destructive part.

The confirmation is an interrupt *before* `delete_reports` runs, so the
assertions here are about what is still in the store while the user is being
asked — not about what the agent said.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.reports import build_report_tools, confirmation_token
from retail_agent.agent.subagents import build_subagents
from retail_agent.agent.supervisor import build_agent


def writer(deps, capture):
    return {fn.__name__: fn for fn in build_subagents(deps, capture)}["report_writer"]


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
    capture = TurnCapture(user_id="exec", session_id="s1", question=question)
    saver = saver or MemorySaver()
    agent = build_agent(deps, capture, checkpointer=saver)
    config = {"configurable": {"thread_id": "s1"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config
    )
    return agent, capture, config, result


def test_a_delete_matching_nothing_never_raises_a_prompt(make_deps, saved):
    """Confirming a no-op is how a confirmation prompt becomes something people
    click through without reading."""
    deps = make_deps(
        script=[[("delete_reports", {"term": "Nothing"})], "I found none."]
    )
    _, _, _, result = run(deps, "delete the reports about Nothing")

    assert not result.get("__interrupt__")
    assert len(saved.list_reports(owner_id="exec")) == 2


def test_nothing_is_deleted_while_the_user_is_being_asked(make_deps, saved):
    """The write cannot have happened: the tool has not run yet."""
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted."]
    )
    _, capture, _, result = run(deps, "delete all reports mentioning Acme")

    assert result.get("__interrupt__")
    assert len(saved.list_reports(owner_id="exec")) == 2
    assert capture.pending is not None
    assert capture.pending.titles == ("Acme Q1",)


def test_approving_deletes_exactly_what_was_shown(make_deps, saved):
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted 1 report."]
    )
    agent, capture, config, _ = run(deps, "delete all reports mentioning Acme")

    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    remaining = [r.title for r in saved.list_reports(owner_id="exec")]
    assert remaining == ["Beta Q1"]


def test_rejecting_leaves_the_library_alone(make_deps, saved):
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Nothing was deleted."]
    )
    agent, _, config, _ = run(deps, "delete all reports mentioning Acme")

    agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config)

    assert len(saved.list_reports(owner_id="exec")) == 2


def test_a_deletion_can_be_undone(make_deps, saved):
    """Soft delete plus an audit row is what makes /undo possible at all."""
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted."]
    )
    agent, _, config, _ = run(deps, "delete all reports mentioning Acme")
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert saved.undo(owner_id="exec") == 1
    assert len(saved.list_reports(owner_id="exec")) == 2


def test_the_manifest_names_every_report_and_the_token_scales(make_deps, saved):
    """One report is a low-stakes correction; several is not."""
    deps = make_deps(script=[[("delete_reports", {"term": ""})], "Deleted."])
    _, capture, _, result = run(deps, "delete all my reports")

    assert set(capture.pending.titles) == {"Acme Q1", "Beta Q1"}
    assert capture.pending.token == confirmation_token(2) == "DELETE 2"

    description = result["__interrupt__"][0].value["action_requests"][0]["description"]
    assert "Acme Q1" in description and "Beta Q1" in description


def test_a_delete_cannot_reach_another_owner(make_deps, saved):
    """Ownership is a WHERE clause in the store, not a check in the agent."""
    deps = make_deps(script=[[("delete_reports", {"term": "Acme"})], "None."])
    capture = TurnCapture(user_id="someone-else", session_id="s1", question="q")
    tools = {fn.__name__: fn for fn in build_report_tools(deps, capture)}

    answer = tools["delete_reports"]("Acme")

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
    capture = TurnCapture(user_id="exec", session_id="s1")

    receipt = writer(deps, capture)(brief="denim findings", title="Q1 Denim")

    assert "Denim fell 11.8%" not in receipt
    assert "Audit Texas" not in receipt
    assert "Q1 Denim" in receipt


def test_what_is_stored_is_what_the_writer_produced(make_deps, reports):
    """No model sits between the two any more, so this is now an identity
    rather than a hope."""
    body = "## Summary\nDenim fell in Q1."
    deps = make_deps(script=[body])
    capture = TurnCapture(user_id="exec", session_id="s1")

    writer(deps, capture)(brief="denim findings", title="Q1 Denim")

    stored = reports.list_reports(owner_id="exec")[0]
    assert stored.body == body
    assert capture.reports_written[0].body == stored.body


def test_a_report_is_scanned_when_it_is_written_not_when_it_is_saved(
    make_deps, reports
):
    """A report shown but never saved used to reach the executive unscanned,
    because the only scan lived inside `save_report`."""
    deps = make_deps(script=["Reach Dana at dana@example.com about this."])
    capture = TurnCapture(user_id="exec", session_id="s1")

    writer(deps, capture)(brief="contacts", title="Contacts")

    assert "dana@example.com" not in capture.reports_written[0].body
    assert "[redacted:email]" in reports.list_reports(owner_id="exec")[0].body


def test_the_show_flag_is_carried_to_the_cli(make_deps):
    """The model decides whether the executive asked to read this; the CLI only
    obeys."""
    deps = make_deps(script=["## Summary\nA draft."])
    capture = TurnCapture(user_id="exec", session_id="s1")

    writer(deps, capture)(brief="b", title="T", show_to_executive=False)

    assert capture.reports_written[0].show is False


def test_a_short_headerless_report_still_leaks_nothing(make_deps):
    """The adversarial shape. An excerpt-based receipt returned this whole body
    verbatim — a report short enough to fit in the receipt is still a report."""
    deps = make_deps(script=["Denim fell 11.8% in Q1. Texas drove it."])
    capture = TurnCapture(user_id="exec", session_id="s1")

    receipt = writer(deps, capture)(brief="b", title="T")

    assert "Denim fell" not in receipt
    assert "Texas" not in receipt


def test_there_is_no_save_report_tool(make_deps):
    """Its only caller was the supervisor retyping a body it had just read."""
    from retail_agent.agent.supervisor import build_tools

    names = {fn.__name__ for fn in build_tools(make_deps(), TurnCapture())}

    assert "save_report" not in names
    assert "report_writer" in names


def asker(deps, capture):
    return {fn.__name__: fn for fn in build_subagents(deps, capture)}["ask_about_report"]


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
    capture = TurnCapture(user_id="exec", session_id="s1")

    answer = asker(deps, capture)(report_id=saved.id, question="what were the actions?")

    assert "Audit Texas" in answer
    assert "Audit Texas inventory depth." in deps.llm.prompts[0]


def test_another_users_report_is_not_readable(make_deps, reports):
    """Ownership is a property of the store's query, not of the caller. A bare
    id must not be enough."""
    saved = reports.save(
        owner_id="someone-else", session_id="s9", title="Theirs", body="Secret."
    )
    deps = make_deps(script=[])
    capture = TurnCapture(user_id="exec", session_id="s1")

    answer = asker(deps, capture)(report_id=saved.id, question="what does it say?")

    assert "Secret" not in answer
    assert "list_reports" in answer


def test_a_missing_report_costs_no_model_call(make_deps):
    """An empty script raises if the subagent is built, so this asserts the
    early return rather than the wording."""
    deps = make_deps(script=[])
    capture = TurnCapture(user_id="exec", session_id="s1")

    answer = asker(deps, capture)(report_id="nope", question="?")

    assert "No report nope" in answer
