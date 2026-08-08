"""What the agent is allowed to remember about the person it is talking to.

`route_node` detected a style preference for free, folded into a routing call
the graph was already making. There is no such call now, so this is elective —
and the rule that made the old detector trustworthy is the one thing that must
not become elective with it: the evidence has to be words the user typed.
"""

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.memory import build_memory_tools
from retail_agent.config import Settings
from retail_agent.obs.traces import InMemoryTraceStore
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.definitions import InMemoryDefinitionStore
from retail_agent.store.learning import InMemorySignalStore
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
        signals=InMemorySignalStore(),
        definitions=InMemoryDefinitionStore(),
    )
    capture = TurnCapture(user_id="dana", session_id="s1", question=question)
    return {fn.__name__: fn for fn in build_memory_tools(deps, capture)}, deps, capture


def test_a_quoted_preference_is_recorded():
    tools, deps, _ = tools_for("just keep it brief, I don't need the workings")

    tools["note_preference"]("depth", "summary", "keep it brief")

    assert deps.signals.counts(user_id="dana") == {
        ("depth", "summary"): (1, "keep it brief")
    }


def test_evidence_the_user_never_typed_is_dropped():
    """The proposal quotes this back — "you asked for this three times, most
    recently '<span>'". A span they never typed makes that a fabrication."""
    tools, deps, _ = tools_for("why are sales down in Texas?")

    answer = tools["note_preference"]("depth", "summary", "keep it brief")

    assert "exact words" in answer
    assert deps.signals.counts(user_id="dana") == {}


def test_a_value_the_setting_does_not_accept_is_refused():
    tools, deps, _ = tools_for("make it snappy")

    answer = tools["note_preference"]("depth", "snappy", "make it snappy")

    assert "not a value" in answer
    assert deps.signals.counts(user_id="dana") == {}


def test_recording_a_signal_never_changes_the_setting():
    """Proposed, never applied. A personalisation the reader cannot account for
    is worse than none."""
    tools, deps, _ = tools_for("keep it brief")

    tools["note_preference"]("depth", "summary", "keep it brief")

    assert deps.preferences.get(user_id="dana").depth == "standard"


def test_a_definition_is_remembered_under_the_term_the_analyst_looks_up():
    """`unresolved` yields lower-cased terms, so a stored 'Loyal' would never be
    found again and the agent would keep asking the same person the same thing."""
    tools, deps, _ = tools_for("loyal means three orders in a year")

    tools["remember_definition"]("Loyal", "three or more orders in a year")

    assert deps.definitions.lookup(user_id="dana", term="loyal") is not None


def test_a_store_failure_costs_the_memory_not_the_turn():
    """The user has just unblocked the agent; failing now wastes that."""
    _, deps, capture = tools_for("loyal means three orders")

    class Broken:
        def remember(self, **_):
            raise RuntimeError("postgres is down")

    object.__setattr__(deps, "definitions", Broken())
    tools = {fn.__name__: fn for fn in build_memory_tools(deps, capture)}

    answer = tools["remember_definition"]("loyal", "three orders")

    assert "could not save" in answer.lower()
