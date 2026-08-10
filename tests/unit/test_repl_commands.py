"""The REPL had no coverage, and a slash command that is handled but not
imported raises NameError only at the moment a user types it — `import app`
succeeds, the whole suite passes, and the command is broken. These exercise the
dispatch loop itself.
"""

import inspect
import io

from retail_agent.knowledge.seeds import SEED_TRIOS

import pytest
from rich.console import Console

from retail_agent.cli.chat import HELP, _repl


class FakeConsole:
    """Answers `input` from a script; renders through a real console.

    Printing `str(arg)` would record "<rich.panel.Panel object at 0x...>" for
    anything boxed, so assertions would pass on output a user cannot read.
    Rendering for real is the only way these tests mean anything.
    """

    def __init__(self, script):
        self.script = list(script)
        self._console = Console(record=True, width=100, file=io.StringIO())

    def input(self, _prompt=""):
        if not self.script:
            raise EOFError
        return self.script.pop(0)

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    @property
    def printed(self) -> list[str]:
        return [line for line in self.text().splitlines() if line.strip()]

    def status(self, *_args, **_kwargs):
        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Null()

    def text(self) -> str:
        # `clear=False` matters: rich empties the record buffer on export, so
        # without it the second assertion in a test reads an empty string and
        # passes or fails for the wrong reason.
        return self._console.export_text(clear=False)


class FakeStore:
    def __init__(self):
        self.undone = 0

    def list_reports(self, *, owner_id, limit=20):
        return []

    def undo(self, *, owner_id, action_id=None):
        self.undone += 1
        return 0


class FakeDeps:
    def __init__(self):
        from retail_agent.obs.traces import InMemoryTraceStore

        from retail_agent.store.personas import (
            DEFAULT_PERSONA,
            InMemoryPersonaStore,
        )

        self.reports = FakeStore()
        self.traces = InMemoryTraceStore()
        from retail_agent.store.preferences import InMemoryPreferenceStore

        self.personas = InMemoryPersonaStore()
        self.personas.seed(DEFAULT_PERSONA)
        self.preferences = InMemoryPreferenceStore()

        from retail_agent.store.definitions import InMemoryDefinitionStore

        self.definitions = InMemoryDefinitionStore()

        from retail_agent.knowledge.trios import InMemoryTrioStore

        self.trios = InMemoryTrioStore(SEED_TRIOS)

        self.settings = type("S", (), {"llm_provider": "gemini"})()


def run(script):
    console = FakeConsole(script)
    deps = FakeDeps()
    code = _repl(console, deps=deps, saver=None, user="dana", session_id="s1")
    return console, deps, code


@pytest.mark.parametrize(
    "command",
    [
        "/help",
        "/reports",
        "/undo",
        "/trace",
        "/trace abc123",
        "/metrics",
        "/persona",
        "/persona list",
        "/persona show",
        "/persona activate analyst",
        "/persona activate nosuchpersona",
        "/prefs",
        "/prefs show_attempt_footnote false",
        "/prefs show_attempt_footnote maybe",
        "/prefs bogus value",
        "/prefs show_attempt_footnote",
        "/trios",
        "/definitions",
        "/definitions forget loyal",
    ],
)
def test_every_command_runs_without_raising(command):
    """A NameError here is invisible to `import app` and to every other test."""
    console, _, code = run([command])

    assert code == 0
    assert console.printed, f"{command} printed nothing"


def test_help_lists_only_commands_the_repl_handles():
    """A command in HELP that the loop does not handle is sent to the agent as
    a question, which is worse than not offering it."""
    advertised = {
        word
        for line in HELP.splitlines()
        for word in line.split()
        if word.startswith("/")
    }

    for command in advertised:
        console, _, _ = run([command])
        assert console.printed, f"{command} is advertised but not handled"


def test_trace_before_any_question_explains_itself():
    console, _, _ = run(["/trace"])

    assert "no turn to trace" in console.text().lower()


def test_trace_by_id_reports_a_miss_rather_than_raising():
    console, _, _ = run(["/trace nosuchturn"])

    assert "no trace for turn" in console.text().lower()


def test_metrics_with_no_turns_says_so():
    console, _, _ = run(["/metrics"])

    assert "no turns recorded" in console.text().lower()


def test_persona_show_names_the_active_one():
    console, _, _ = run(["/persona show"])

    assert "analyst" in console.text().lower()


def test_activating_an_unknown_persona_says_so_rather_than_raising():
    console, deps, _ = run(["/persona activate nosuchpersona"])

    assert "no persona" in console.text().lower()
    assert deps.personas.active().name == "analyst", "unchanged"


def test_activating_switches_the_voice():
    console = FakeConsole(["/persona activate terse"])
    deps = FakeDeps()
    deps.personas.save(name="terse", body="Be brief.", updated_by="ceo")

    _repl(console, deps=deps, saver=None, user="dana", session_id="s1")

    assert deps.personas.active().name == "terse"


def test_prefs_change_is_persisted():
    console, deps, _ = run(["/prefs show_attempt_footnote false"])

    assert deps.preferences.get(user_id="dana").show_attempt_footnote is False
    assert "false" in console.text().lower()


def test_a_rejected_pref_value_explains_the_options_and_changes_nothing():
    console, deps, _ = run(["/prefs show_attempt_footnote maybe"])

    assert "true" in console.text().lower(), "the options are named"
    assert deps.preferences.get(user_id="dana").show_attempt_footnote is True, (
        "unchanged"
    )


def test_a_retired_setting_is_named_as_unknown():
    """`answer_format` and friends were stored and read by nothing. Offering
    them would be a settings screen that silently does nothing."""
    console, deps, _ = run(["/prefs answer_format bullets"])

    assert "unknown" in console.text().lower()


def test_prefs_are_per_user():
    """Manager A's rendering choice must never leak into Manager B's."""
    from retail_agent.store.preferences import InMemoryPreferenceStore

    shared = InMemoryPreferenceStore()
    for user, value in (("dana", "false"), ("sam", "true")):
        console = FakeConsole([f"/prefs show_attempt_footnote {value}"])
        deps = FakeDeps()
        deps.preferences = shared
        _repl(console, deps=deps, saver=None, user=user, session_id="s1")

    assert shared.get(user_id="dana").show_attempt_footnote is False
    assert shared.get(user_id="sam").show_attempt_footnote is True


# --- the definition prompt ---


def _answer(script, term="loyal customers", options=("three or more orders",)):
    from retail_agent.cli.chat import _ask_definition

    console = FakeConsole(script)
    return _ask_definition(console, term, list(options)), console


def test_a_slash_command_is_not_recorded_as_a_definition():
    """Observed live. At this prompt the executive typed `/persona`, meaning to
    run the command, and it was saved as the meaning of "loyal customers".

    Every later turn then read that back: the tool reported the term settled,
    handed the model "/persona" as its definition, and the model — correctly
    judging that meaningless — asked all over again. A junk definition is worse
    than none, because `/definitions forget` is the only way out and nobody
    knows there is anything to forget.

    The reasoning is the one already in this function's docstring for digits:
    someone who types a command meant to run it, not to define a word.
    """
    chosen, console = _answer(["/persona", "1"])

    assert chosen == "three or more orders", "the re-ask was answered with 1"
    assert any("command" in line.lower() for line in console.printed), (
        "the user is told why their command did not take"
    )


def test_a_real_definition_starting_with_a_word_is_still_accepted():
    chosen, _ = _answer(["customers with three or more completed orders"])

    assert chosen == "customers with three or more completed orders"


def test_a_slash_command_in_the_write_your_own_path_is_not_recorded():
    """The same incident, through the adjacent door. Picking "write my own"
    and then typing `/persona` must re-ask exactly as the first prompt does —
    a guard that covers one of two input paths only moves the bug."""
    chosen, console = _answer(["2", "/persona", "five or more orders"])

    assert chosen == "five or more orders"
    assert any("command" in line.lower() for line in console.printed)


def test_a_command_full_of_brackets_neither_crashes_nor_vanishes():
    """The warning echoes what was typed. Unescaped, `[/]` parses as a Rich
    closing tag and raises MarkupError mid-interrupt — killing the resume flow
    instead of re-asking — and `[loyal]` parses as a style tag and silently
    disappears from the message."""
    chosen, console = _answer(["/definitions forget [loyal]", "1"])

    assert chosen == "three or more orders"
    assert any("[loyal]" in line for line in console.printed), (
        "the command is echoed back as typed"
    )

    chosen, _ = _answer(["/x [/]", "1"])

    assert chosen == "three or more orders"
