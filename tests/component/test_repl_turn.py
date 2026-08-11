"""The REPL driving a real turn, including the confirmation flow.

The interrupt payload shape is the reason this file exists. `Command(resume=...)`
takes a dict the middleware subscripts by name; a bare list type-checks, imports,
and fails only at the moment a user confirms a deletion — which is the worst
possible moment to find out.
"""

import io

import pytest
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from retail_agent.cli.chat import _answer


class FakeConsole:
    """Answers `input` from a script; renders through a real console."""

    def __init__(self, script=()):
        self.script = list(script)
        self._console = Console(record=True, width=100, file=io.StringIO())

    def input(self, _prompt=""):
        if not self.script:
            raise EOFError
        return self.script.pop(0)

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def status(self, *_args, **_kwargs):
        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Null()

    def text(self) -> str:
        return self._console.export_text(clear=False)


@pytest.fixture
def saved(reports):
    reports.save(owner_id="dana", session_id="s1", title="Acme Q1", body="Acme.")
    reports.save(owner_id="dana", session_id="s1", title="Beta Q1", body="Beta.")
    return reports


def answer(console, deps, question):
    return _answer(console, deps, MemorySaver(), "dana", "s1", question)


def test_an_ordinary_turn_renders_and_returns_its_trace(make_deps):
    console = FakeConsole()
    deps = make_deps(script=["Revenue was 12."])

    trace = answer(console, deps, "what was revenue?")

    assert "Revenue was 12." in console.text()
    assert trace is not None and trace.owner_id == "dana"


def test_typing_the_token_confirms_the_deletion(make_deps, saved):
    console = FakeConsole(["y"])
    deps = make_deps(
        script=[[("delete_reports", {"term": "Acme"})], "Deleted 1 report."]
    )

    answer(console, deps, "delete the reports mentioning Acme")

    assert [r.title for r in saved.list_reports(owner_id="dana")] == ["Beta Q1"]
    assert "Acme Q1" in console.text(), "the manifest named what would go"


def test_typing_anything_else_cancels(make_deps, saved):
    """"Anything else cancels" has to include a plausible near-miss: someone who
    types `yes` when the token is `DELETE 2` has not read the manifest."""
    console = FakeConsole(["yes"])
    deps = make_deps(script=[[("delete_reports", {"term": ""})], "Nothing deleted."])

    answer(console, deps, "delete all my reports")

    assert len(saved.list_reports(owner_id="dana")) == 2
    assert "Cancelled" in console.text()


def test_a_bulk_delete_needs_the_counted_token(make_deps, saved):
    console = FakeConsole(["DELETE 2"])
    deps = make_deps(script=[[("delete_reports", {"term": ""})], "Deleted 2 reports."])

    answer(console, deps, "delete all my reports")

    assert saved.list_reports(owner_id="dana") == []


def _failing(make_deps):
    """A provider-shaped failure rather than `ScriptExhausted`, which is a
    `BaseException` on purpose so an over-broad `except Exception` in the code
    under test cannot swallow a test's own bug."""
    deps = make_deps(script=["unused"])

    def explode(*_args, **_kwargs):
        raise RuntimeError("the provider is down")

    object.__setattr__(deps.llm, "_generate", explode)
    return deps


def test_a_failing_turn_is_rendered_rather_than_raised(make_deps):
    """The REPL must survive anything: an exception here ends the session and
    loses the conversation."""
    console = FakeConsole()

    answer(console, _failing(make_deps), "what was revenue?")

    assert "went wrong" in console.text()


def test_a_failing_turn_still_leaves_a_trace(make_deps, traces):
    """The recorder is an `after_agent` hook, so it never runs when the agent
    raises. The turn id printed on the error panel then resolved to nothing, and
    `/trace` went on showing whichever turn last succeeded."""
    console = FakeConsole()

    trace = answer(console, _failing(make_deps), "what was revenue?")

    assert trace is not None, "/trace must describe the turn that just failed"
    assert trace.question == "what was revenue?"
    assert trace.status == "failed"

    printed = console.text()
    assert trace.turn_id in printed, "the id on the error panel has to resolve"
    assert traces.get(owner_id="dana", turn_id=trace.turn_id) is not None


def test_the_answer_reaches_the_console_exactly_once(make_deps):
    """`_stream_turn` prints the answer live as it arrives; `render_answer`
    used to print the same text again afterward through `Markdown`, so it
    appeared twice. This pins "exactly once" — the property this fake can
    actually verify.

    It does NOT pin that the answer arrives incrementally rather than in one
    shot: `ScriptedChatModel` returns a whole message per turn, not token
    chunks, so this fake cannot show the difference between streaming and
    printing a finished string. That property is pinned by Task 3's boundary
    test (`test_progress_from_inside_the_analyst_reaches_the_caller`), which
    fails if the analyst goes back to invoking its subagent instead of
    streaming it. Do not re-add a containment assertion here believing it
    proves streaming — it would pass just as well against the old blocking
    `_answer`.
    """
    console = FakeConsole()
    deps = make_deps(script=["Denim fell 11.8% in Q1."])

    answer(console, deps, "how did denim do?")

    assert console.text().count("Denim fell 11.8%") == 1


def test_a_failed_turn_prints_no_report(make_deps, monkeypatch):
    """The error panel is what the executive gets. A report printed underneath
    it would claim the turn produced something the turn did not finish."""
    import io

    from rich.console import Console

    from retail_agent.cli import chat as chat_module

    console = Console(record=True, width=100, file=io.StringIO())
    deps = make_deps(script=[])

    def explode(*args, **kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(chat_module, "build_agent", explode)

    chat_module._answer(console, deps, None, "exec", "s1", "write me a report")

    printed = console.export_text()
    assert "Something went wrong" in printed
    assert "Saved as" not in printed
