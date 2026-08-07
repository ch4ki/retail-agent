"""`_answer` — the part of the REPL that drives a turn.

Untested until a NameError in the definition-pause branch reached a user. The
existing REPL tests run with `graph=None`, so they exercise dispatch and never
reach either pause loop; `import chat` cannot catch it either, because a missing
name only fires when that branch runs.

These use a fake graph that reports pauses the way LangGraph does, so the two
breakpoint paths and the error path all execute.
"""

import io

import pytest
from rich.console import Console

from retail_agent.cli.chat import _answer
from retail_agent.store.definitions import InMemoryDefinitionStore
from retail_agent.store.preferences import InMemoryPreferenceStore


class Snapshot:
    def __init__(self, nxt, values):
        self.next = nxt
        self.values = values


class FakeGraph:
    """Reports a scripted sequence of pauses, then finishes.

    `next` is a tuple naming the node the graph is waiting before — exactly what
    `graph.get_state(config).next` returns after `interrupt_before` fires.
    """

    def __init__(self, pauses=(), final=None):
        self.pauses = list(pauses)
        self.final = final or {"answer": "Done.", "turn_id": "t1"}
        self.resumed_with = []
        self._snapshot = self._next_snapshot()

    def _next_snapshot(self):
        if self.pauses:
            node, values = self.pauses[0]
            return Snapshot((node,), values)
        return Snapshot((), self.final)

    def get_state(self, _config):
        return self._snapshot

    def update_state(self, _config, values, as_node=None):
        self.resumed_with.append((as_node, values))
        self.pauses.pop(0)
        self._snapshot = self._next_snapshot()

    def invoke(self, _input, config=None):  # `config=` keyword, as run_turn calls it
        return self._snapshot.values


class RecordingConsole:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self._console = Console(record=True, width=100, file=io.StringIO())

    def input(self, _prompt=""):
        return self.replies.pop(0) if self.replies else ""

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def status(self, *_a, **_k):
        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Null()

    def text(self):
        return self._console.export_text(clear=False)


class FakeDeps:
    def __init__(self):
        self.settings = type("S", (), {"llm_provider": "gemini"})()
        self.preferences = InMemoryPreferenceStore()
        self.definitions = InMemoryDefinitionStore()
        self.traces = type("T", (), {"record": lambda self, trace: None})()


def drive(graph, replies=()):
    console = RecordingConsole(replies)
    deps = FakeDeps()
    state = _answer(console, graph, deps, "dana", "s1", "how many at risk?")
    return console, deps, state


def test_a_plain_turn_renders_the_answer():
    console, _, state = drive(FakeGraph(final={"answer": "42 customers.", "turn_id": "t1"}))

    assert "42 customers." in console.text()
    assert state["answer"] == "42 customers."


def test_a_definition_pause_asks_and_resumes():
    """The branch that shipped broken: a NameError here is invisible to every
    other test and to `import chat`."""
    graph = FakeGraph(
        pauses=[("await_definition", {"pending_term": "at risk"})],
        final={"answer": "85 customers.", "turn_id": "t1"},
    )

    console, _, _ = drive(graph, replies=["no order in 120 days"])

    assert "at risk" in console.text(), "the user was told which term"
    assert graph.resumed_with == [
        ("await_definition", {"definition_reply": "no order in 120 days"})
    ]
    assert "85 customers." in console.text()


def test_pressing_enter_at_a_definition_pause_still_resumes():
    graph = FakeGraph(
        pauses=[("await_definition", {"pending_term": "loyal"})],
        final={"answer": "Assumed 3+ orders.", "turn_id": "t1"},
    )

    _, _, _ = drive(graph, replies=[""])

    assert graph.resumed_with == [("await_definition", {"definition_reply": ""})]


def test_a_confirmation_pause_shows_the_manifest_and_resumes():
    from retail_agent.agent.state import PendingAction

    action = PendingAction(
        action_id="a1", report_ids=("r1", "r2"), titles=("Q1", "Q2"), token="DELETE 2"
    )
    graph = FakeGraph(
        pauses=[("await_confirmation", {"pending_action": action})],
        final={"answer": "Deleted 2.", "turn_id": "t1"},
    )

    console, _, _ = drive(graph, replies=["DELETE 2"])

    assert "Q1" in console.text() and "DELETE 2" in console.text()
    assert graph.resumed_with == [("await_confirmation", {"confirmation": "DELETE 2"})]


def test_both_pauses_in_one_turn_are_each_handled():
    """A delete is not the only thing that can pause, and a turn could hit an
    undefined term and then a confirmation."""
    from retail_agent.agent.state import PendingAction

    action = PendingAction(
        action_id="a1", report_ids=("r1",), titles=("Q1",), token="y"
    )
    graph = FakeGraph(
        pauses=[
            ("await_definition", {"pending_term": "at risk"}),
            ("await_confirmation", {"pending_action": action}),
        ],
        final={"answer": "Done.", "turn_id": "t1"},
    )

    _, _, _ = drive(graph, replies=["no order in 120 days", "y"])

    assert [node for node, _ in graph.resumed_with] == [
        "await_definition",
        "await_confirmation",
    ]


def test_a_failing_turn_renders_an_error_rather_than_raising():
    class Exploding(FakeGraph):
        def invoke(self, _input, config=None):  # `config=` keyword, as run_turn calls it
            raise RuntimeError("429 rate limited")

    console, _, _ = drive(Exploding())

    assert "quota" in console.text().lower() or "429" in console.text()


def test_the_repl_does_not_record_the_turn_itself():
    """Recording moved into the graph's finish_turn node, so every caller gets a
    trace rather than only this one. A second write here would double-count
    every turn in `/metrics`."""
    recorded = []

    class Deps(FakeDeps):
        def __init__(self):
            super().__init__()
            self.traces = type(
                "T", (), {"record": lambda _self, trace: recorded.append(trace)}
            )()

    console = RecordingConsole()
    _answer(console, FakeGraph(), Deps(), "dana", "s1", "how many at risk?")

    assert recorded == []
