"""What the agent is allowed to remember about the person it is talking to.

The proposal engine this replaced accumulated signals in Postgres and asked at
three sightings, because it sat on top of a *guess* at intent — first a regex,
then a classifier. The honest response to a guess is to propose it rather than
act on it.

This tool is not a guess: it fires only on words the user typed, checked here
against the recorded question. So the rule that must survive is the
quotable-evidence check. Without it the tool writes an inference as though it
were an instruction, which is the failure the proposal engine existed to
prevent.
"""

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.memory import build_memory_tools
from retail_agent.config import Settings
from retail_agent.obs.traces import InMemoryTraceStore
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.definitions import InMemoryDefinitionStore
from retail_agent.store.memory_reports import InMemoryReportStore
from retail_agent.store.preferences import InMemoryPreferenceStore


def tools_for(question):
    deps = AgentDeps(
        settings=Settings(_env_file=None, google_cloud_project="test"),
        llm=object(),
        source=object(),
        policy=PiiPolicy.default(),
        reports=InMemoryReportStore(),
        traces=InMemoryTraceStore(),
        preferences=InMemoryPreferenceStore(),
        definitions=InMemoryDefinitionStore(),
    )
    capture = TurnCapture(user_id="dana", session_id="s1", question=question)
    return {fn.__name__: fn for fn in build_memory_tools(deps, capture)}, deps, capture


def test_a_quoted_preference_is_saved_in_the_users_own_words():
    tools, deps, _ = tools_for("just keep it brief, I don't need the workings")

    tools["note_preference"]("keep answers brief", "keep it brief")

    assert deps.preferences.list_notes(user_id="dana") == ["keep answers brief"]


def test_the_change_is_recorded_so_the_interface_can_announce_it():
    """Applying immediately is defensible; applying *silently* is not. The CLI
    reads this rather than trusting the model to mention what it changed."""
    tools, _, capture = tools_for("keep it brief")

    tools["note_preference"]("keep answers brief", "keep it brief")

    assert capture.preference_changes == [("added", "keep answers brief")]


def test_evidence_the_user_never_typed_changes_nothing():
    """The check that separates "they asked for this" from "the model thinks
    they want this" — and this question is about the data, not the layout.

    This guard is the whole reason the tool may act rather than propose, so it
    outlived the enum validation that used to sit above it."""
    tools, deps, capture = tools_for("why are sales down in Texas?")

    answer = tools["note_preference"]("keep answers brief", "keep it brief")

    assert "exact words" in answer
    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes == []


def test_a_preference_already_saved_is_not_saved_twice():
    tools, deps, capture = tools_for("keep it brief")
    tools["note_preference"]("keep answers brief", "keep it brief")

    answer = tools["note_preference"]("Keep answers brief", "keep it brief")

    assert "already" in answer.lower()
    assert deps.preferences.list_notes(user_id="dana") == ["keep answers brief"]
    assert capture.preference_changes == [("added", "keep answers brief")], "announced once"


def test_a_preference_past_the_cap_is_refused_with_a_way_out():
    from retail_agent.store.preferences import MAX_NOTES

    tools, deps, capture = tools_for("keep it brief")
    deps.preferences.replace_notes(
        user_id="dana", notes=[f"preference {i}" for i in range(MAX_NOTES)]
    )

    answer = tools["note_preference"]("keep answers brief", "keep it brief")

    assert "forget" in answer.lower(), "says how to make room"
    assert len(deps.preferences.list_notes(user_id="dana")) == MAX_NOTES
    assert capture.preference_changes == []


def test_an_over_long_preference_is_refused_rather_than_cut_down():
    """A truncated note is a preference the user did not write."""
    from retail_agent.store.preferences import MAX_NOTE_CHARS

    tools, deps, capture = tools_for("keep it brief")

    answer = tools["note_preference"]("x" * (MAX_NOTE_CHARS + 1), "keep it brief")

    assert "shorter" in answer.lower()
    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes == []


def test_a_store_failure_costs_the_preference_not_the_turn():
    """Losing a preference must not lose the answer it was about."""
    _, deps, capture = tools_for("keep it brief")

    class Broken:
        def list_notes(self, **_):
            raise RuntimeError("postgres is down")

        def replace_notes(self, **_):
            raise RuntimeError("postgres is down")

    object.__setattr__(deps, "preferences", Broken())
    tools = {fn.__name__: fn for fn in build_memory_tools(deps, capture)}

    answer = tools["note_preference"]("keep answers brief", "keep it brief")

    assert "could not save" in answer.lower()
    assert capture.preference_changes == []


def test_forgetting_removes_the_note_and_announces_it():
    tools, deps, capture = tools_for("stop showing prices in euros")
    tools["note_preference"]("show prices in euros", "prices in euros")

    answer = tools["forget_preference"]("show prices in euros")

    assert "removed" in answer.lower()
    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes[-1] == ("removed", "show prices in euros")


def test_forgetting_something_that_was_never_saved_says_so():
    tools, deps, capture = tools_for("stop showing prices in euros")

    answer = tools["forget_preference"]("show prices in euros")

    assert "nothing" in answer.lower()
    assert capture.preference_changes == []


def test_editing_a_preference_is_a_forget_then_a_note():
    """There is no edit tool, so this is the path the model has to take —
    and both halves have to reach the announcement."""
    tools, deps, capture = tools_for("make that under two sentences, not three")
    deps.preferences.replace_notes(user_id="dana", notes=["keep answers under three sentences"])

    tools["forget_preference"]("keep answers under three sentences")
    tools["note_preference"]("keep answers under two sentences", "under two sentences")

    assert deps.preferences.list_notes(user_id="dana") == [
        "keep answers under two sentences"
    ]
    assert capture.preference_changes == [
        ("removed", "keep answers under three sentences"),
        ("added", "keep answers under two sentences"),
    ]


def test_a_definition_is_remembered_under_the_term_the_analyst_looks_up():
    """`unresolved` yields lower-cased terms, so a stored 'Loyal' would never be
    found again and the agent would keep asking the same person the same thing."""
    tools, deps, _ = tools_for("loyal means three orders in a year")

    tools["remember_definition"]("Loyal", "three or more orders in a year")

    assert deps.definitions.lookup(user_id="dana", term="loyal") is not None


def test_a_definition_store_failure_costs_the_memory_not_the_turn():
    """The user has just unblocked the agent; failing now wastes that."""
    _, deps, capture = tools_for("loyal means three orders")

    class Broken:
        def remember(self, **_):
            raise RuntimeError("postgres is down")

    object.__setattr__(deps, "definitions", Broken())
    tools = {fn.__name__: fn for fn in build_memory_tools(deps, capture)}

    answer = tools["remember_definition"]("loyal", "three orders")

    assert "could not save" in answer.lower()
