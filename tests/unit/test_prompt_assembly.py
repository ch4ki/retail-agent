"""What reaches the model, assembled per call.

The middleware object `@dynamic_prompt` returns takes a `ModelRequest` and is
awkward to call directly, so the assembly is a plain function underneath it and
this is a test of that function. The decorator is then one line with nothing in
it worth testing.
"""

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.middleware import supervisor_system_prompt
from retail_agent.config import Settings
from retail_agent.obs.traces import InMemoryTraceStore
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.memory_reports import InMemoryReportStore
from retail_agent.store.preferences import InMemoryPreferenceStore, add_note


def deps_with(preferences):
    return AgentDeps(
        settings=Settings(_env_file=None, google_cloud_project="test"),
        llm=object(),
        source=object(),
        policy=PiiPolicy.default(),
        reports=InMemoryReportStore(),
        traces=InMemoryTraceStore(),
        preferences=preferences,
    )


def capture_for(user_id="dana"):
    return TurnCapture(user_id=user_id, session_id="s1", question="how are sales?")


def test_the_supervisor_prompt_carries_the_users_notes():
    """Assembled per call, so a note saved mid-session reaches the next model
    call rather than the next restart."""
    preferences = InMemoryPreferenceStore()
    add_note(preferences, user_id="dana", note="show prices in euros")

    prompt = supervisor_system_prompt(deps_with(preferences), capture_for())

    assert "show prices in euros" in prompt


def test_a_user_with_no_notes_gets_no_empty_heading():
    """A heading with nothing under it is noise the model has to read past."""
    prompt = supervisor_system_prompt(
        deps_with(InMemoryPreferenceStore()), capture_for()
    )

    assert "has asked for" not in prompt
    assert not prompt.endswith("\n"), "no dangling separator where the block was"


def test_one_users_notes_never_reach_another_users_prompt():
    preferences = InMemoryPreferenceStore()
    add_note(preferences, user_id="dana", note="show prices in euros")

    prompt = supervisor_system_prompt(deps_with(preferences), capture_for("sam"))

    assert "euros" not in prompt


def test_a_broken_store_still_produces_a_prompt():
    """Never fail a turn over a preference."""

    class Broken:
        def list_notes(self, **_):
            raise RuntimeError("postgres is down")

    prompt = supervisor_system_prompt(deps_with(Broken()), capture_for())

    assert prompt, "the turn goes ahead without the notes"
