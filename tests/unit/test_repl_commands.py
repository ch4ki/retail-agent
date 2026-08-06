"""The REPL had no coverage, and a slash command that is handled but not
imported raises NameError only at the moment a user types it — `import app`
succeeds, the whole suite passes, and the command is broken. These exercise the
dispatch loop itself.
"""

import pytest

from retail_agent.cli.app import HELP, _repl


class FakeConsole:
    """Answers `input` from a script and records everything printed."""

    def __init__(self, script):
        self.script = list(script)
        self.printed: list[str] = []

    def input(self, _prompt=""):
        if not self.script:
            raise EOFError
        return self.script.pop(0)

    def print(self, *args, **kwargs):
        self.printed.append(" ".join(str(a) for a in args))

    def status(self, *_args, **_kwargs):
        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Null()

    def text(self) -> str:
        return "\n".join(self.printed)


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

        self.reports = FakeStore()
        self.traces = InMemoryTraceStore()
        self.settings = type("S", (), {"llm_provider": "gemini"})()


def run(script):
    console = FakeConsole(script)
    deps = FakeDeps()
    code = _repl(console, graph=None, deps=deps, user="dana", session_id="s1")
    return console, deps, code


@pytest.mark.parametrize(
    "command",
    ["/help", "/reports", "/undo", "/trace", "/trace abc123", "/metrics"],
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


def test_undo_reaches_the_store():
    _, deps, _ = run(["/undo"])

    assert deps.reports.undone == 1


def test_quit_exits_cleanly():
    console, _, code = run(["/quit"])

    assert code == 0
    assert "bye" in console.text().lower()


def test_blank_input_is_ignored_not_sent_to_the_agent():
    # graph=None, so any attempt to answer would raise.
    console, _, code = run(["", "   ", "/quit"])

    assert code == 0
