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


def test_a_quoted_preference_becomes_the_default():
    tools, deps, _ = tools_for("just keep it brief, I don't need the workings")

    tools["note_preference"]("depth", "summary", "keep it brief")

    assert deps.preferences.get(user_id="dana").depth == "summary"


def test_the_change_is_recorded_so_the_interface_can_announce_it():
    """Applying immediately is defensible; applying *silently* is not. The CLI
    reads this rather than trusting the model to mention what it changed."""
    tools, _, capture = tools_for("keep it brief")

    tools["note_preference"]("depth", "summary", "keep it brief")

    assert capture.preference_changes == [("depth", "summary")]


def test_evidence_the_user_never_typed_changes_nothing():
    """The check that separates "they asked for this" from "the model thinks
    they want this" — and this question is about the data, not the layout."""
    tools, deps, capture = tools_for("why are sales down in Texas?")

    answer = tools["note_preference"]("depth", "summary", "keep it brief")

    assert "exact words" in answer
    assert deps.preferences.get(user_id="dana").depth == "standard"
    assert capture.preference_changes == []


def test_a_value_the_setting_does_not_accept_is_refused():
    tools, deps, capture = tools_for("make it snappy")

    answer = tools["note_preference"]("depth", "snappy", "make it snappy")

    assert "not a value" in answer
    assert deps.preferences.get(user_id="dana").depth == "standard"
    assert capture.preference_changes == []


def test_a_store_failure_costs_the_default_not_the_turn():
    """Losing a layout setting must not lose the answer it was about."""
    _, deps, capture = tools_for("keep it brief")

    class Broken:
        def get(self, **_):
            raise RuntimeError("postgres is down")

        def set(self, **_):
            raise RuntimeError("postgres is down")

    object.__setattr__(deps, "preferences", Broken())
    tools = {fn.__name__: fn for fn in build_memory_tools(deps, capture)}

    answer = tools["note_preference"]("depth", "summary", "keep it brief")

    assert "could not save" in answer.lower()
    assert capture.preference_changes == []


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
