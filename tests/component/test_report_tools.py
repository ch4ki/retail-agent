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
from retail_agent.agent.supervisor import build_agent


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


def test_saving_scans_the_body_before_it_becomes_durable(make_deps, reports):
    """A report is read long after the conversation, by people who were not in
    it. This is the one copy nobody will double-check."""
    deps = make_deps()
    tools = {fn.__name__: fn for fn in build_report_tools(deps, TurnCapture(user_id="exec"))}

    tools["save_report"]("Contacts", "Reach Dana at dana@example.com about this.")

    stored = reports.list_reports(owner_id="exec")[0]
    assert "dana@example.com" not in stored.body
    assert "[redacted:email]" in stored.body


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
