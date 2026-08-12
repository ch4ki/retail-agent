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

from retail_agent.agent.subagents import final_text
from retail_agent.agent.supervisor import build_agent
from retail_agent.cli.chat import _answer

from .conftest import StreamingScriptedChatModel


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

    This also only covers the simple path: no tool here makes a model call of
    its own. `test_a_report_writers_draft_does_not_leak_into_the_answer` and
    `test_ask_about_reports_internal_reply_does_not_leak_into_the_answer`
    below cover the path where a tool's own internal `model.invoke()` would
    otherwise surface in the same stream — see `_SUPERVISOR_NODE` in
    `chat.py` for why that happens and how it is filtered out.
    """
    console = FakeConsole()
    deps = make_deps(script=["Denim fell 11.8% in Q1."])

    answer(console, deps, "how did denim do?")

    assert console.text().count("Denim fell 11.8%") == 1


def test_the_streamed_answer_matches_what_the_turn_recorded(make_deps):
    """I3: no test in this branch used a model that actually streams —
    `ScriptedChatModel` only implements `_generate`, so `.stream()` fell back
    to handing `_stream_turn` one whole-message chunk, which cannot show
    either of two real bugs: per-chunk `.strip()` deleting every space
    between words (C1), and every `model`-node message getting glued into
    one answer instead of just the final one (C2).

    `StreamingScriptedChatModel` streams for real, word by word, and the turn
    below narrates alongside two tool calls before its real answer — the
    shape the review measured as producing
    `'Let me look that up.Now writing it up.Revenuewas12percentinQ1.'`
    from `_stream_turn` while the checkpoint recorded only the final
    sentence. This is the single assertion that catches both: what the
    executive read on screen must be exactly what the turn recorded.
    """
    console = FakeConsole()
    saver = MemorySaver()
    deps = make_deps(
        llm=StreamingScriptedChatModel(
            [
                ("Let me look that up.", [("list_reports", {})]),
                ("Now writing it up.", [("list_reports", {})]),
                "Revenue was 12 percent in Q1.",
            ]
        )
    )
    config = {"configurable": {"thread_id": "s1"}}

    trace = _answer(console, deps, saver, "dana", "s1", "how did we do?")

    recorded_agent = build_agent(deps, checkpointer=saver)
    recorded = final_text(recorded_agent.get_state(config).values)

    assert trace.answer == recorded
    assert trace.answer == "Revenue was 12 percent in Q1."


def test_bracketed_answer_text_survives_to_the_console(make_deps):
    """C3, silent-loss half: `console.print(text, end="")` interprets Rich
    console markup, and the model's own words are not console markup.
    `render_answer` never had this problem — it printed `Markdown(answer)`,
    which does not parse `[...]` at all — but streaming per token lost that
    for free. Measured before the fix: `'Sales fell [see appendix] and
    margins [/] held.'` printed as `'Sales fell  and margins  held.'`, the
    bracketed spans silently swallowed as markup rather than shown.
    """
    console = FakeConsole()
    deps = make_deps(
        llm=StreamingScriptedChatModel(
            ["Sales fell [see appendix] and margins [/] held."]
        )
    )

    trace = answer(console, deps, "how did sales do?")

    assert "Sales fell [see appendix] and margins [/] held." in console.text()
    assert trace.answer == "Sales fell [see appendix] and margins [/] held."


def test_an_unmatched_markup_tag_in_the_answer_does_not_kill_the_turn(make_deps):
    """C3, hard-failure half: an unmatched closing tag such as `[/bold]` used
    to raise `rich.errors.MarkupError` straight out of `console.print`, and
    that escaped `_answer`'s own exception handling — `_stream_turn` is
    called from inside `_answer`'s `try`, but the print happens while the
    stream is still open, so the raised `MarkupError` propagated up through
    `_answer` itself and out to `_repl`, which has no `try` around the call
    (`chat.py:275`). The whole session died instead of the turn failing,
    voiding `_answer`'s own "always returns its trace" contract.
    """
    console = FakeConsole()
    deps = make_deps(
        llm=StreamingScriptedChatModel(["Margins [/bold] held and costs [red]rose."])
    )

    trace = answer(console, deps, "how did margins do?")

    assert trace is not None
    assert "Margins [/bold] held and costs [red]rose." in console.text()


def test_the_matched_count_does_not_reprint_after_a_cancelled_delete(make_deps, saved):
    """I2: `delete_reports` resolves the match count and used to announce it
    with `runtime.stream_writer` before pausing for confirmation. The whole
    tool body replays from the top on resume (see the comment above `pending`
    in `reports.py`), and `interrupt()` only pauses the first time through —
    so the announcement fired again on the resuming call, after the
    executive had already typed something other than the token. Measured
    before the fix: "1 report(s) matched" printed once before the
    confirmation panel, then "Cancelled — nothing was deleted.", then "1
    report(s) matched" again — which could read as the deletion having gone
    ahead despite the cancellation.
    """
    console = FakeConsole(["no"])
    deps = make_deps(script=[[("delete_reports", {"term": ""})], "Nothing deleted."])

    answer(console, deps, "delete all my reports")

    printed = console.text()
    assert printed.count("report(s) matched") == 1
    assert "Cancelled" in printed
    assert len(saved.list_reports(owner_id="dana")) == 2


def test_the_node_guard_fires_when_the_supervisor_never_appears(make_deps):
    """The `RuntimeError` guard at the end of `_stream_turn` exists for one
    reason: if `_SUPERVISOR_NODE` is ever wrong — the graph's model node
    renamed out from under it — silently rendering nothing would be a worse
    bug than a loud failure. No turn through the real, correctly-named graph
    can trigger it: the supervisor always runs its own node at least once,
    so `seen_nodes` always contains it whenever any `AIMessage` chunk is
    seen at all. That makes the guard untestable through a legitimate turn,
    which is exactly why it shipped with no test — this drives `_stream_turn`
    directly against a stub `agent.stream()` that only ever yields an
    `AIMessage` chunk from some other node, standing in for the rename this
    guard is meant to catch.
    """
    from langchain_core.messages import AIMessage

    from retail_agent.cli.chat import _stream_turn

    class _RenamedNodeAgent:
        def stream(self, *_args, **_kwargs):
            yield "messages", (
                AIMessage(content="leaked"),
                {"langgraph_node": "some_other_node"},
            )

    with pytest.raises(RuntimeError, match="supervisor's node name changed"):
        _stream_turn(FakeConsole(), _RenamedNodeAgent(), {}, {}, None)


def test_a_report_writers_draft_does_not_leak_into_the_answer(make_deps):
    """`report_writer` drafts the report with a bare `model.invoke()` inside
    `resilient_call`. LangGraph propagates the parent run's callback context
    into that call, so the draft surfaces as an `AIMessage` chunk on
    `stream_mode="messages"` too — tagged `langgraph_node="tools"`, never
    touching `TurnState["messages"]`, which is why `final_text(result)` never
    saw it before streaming existed.

    Unfiltered, `_stream_turn` glued that draft onto the covering sentence
    with no separator, and — because `_stream_turn`'s return value becomes
    `trace.answer` — persisted the glued text to the trace, not just the
    console. `/trace` and every later audit would have shown it.
    """
    console = FakeConsole()
    deps = make_deps(
        script=[
            [("report_writer", {"brief": "denim", "title": "Q1 Report"})],
            "# Q1 Report\n\nRevenue up 5%.",
            "Here's your report.",
        ]
    )

    trace = answer(console, deps, "write me a report on denim")

    assert trace.answer == "Here's your report."


def test_ask_about_reports_internal_reply_does_not_leak_into_the_answer(
    make_deps, reports
):
    """Same leak, a different tool: `ask_about_report` also drafts its reply
    with a bare `model.invoke()` inside `resilient_call`, and that call's
    `AIMessage` chunk is tagged `langgraph_node="tools"` for the same reason.
    Unfiltered, the internal reply and the supervisor's own covering sentence
    both rendered — the answer appeared twice, once from each node.
    """
    saved_report = reports.save(
        owner_id="dana", session_id="s1", title="Q1 Denim", body="Denim rose 3% in Q1."
    )
    console = FakeConsole()
    deps = make_deps(
        script=[
            [
                (
                    "ask_about_report",
                    {"report_id": saved_report.id, "question": "how did denim do?"},
                )
            ],
            "Denim rose 3%, driven by wholesale.",
            "It rose 3%, mostly wholesale.",
        ]
    )

    trace = answer(console, deps, "what does the denim report say?")

    assert trace.answer == "It rose 3%, mostly wholesale."


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


def test_the_wait_is_shown_until_the_first_output(make_deps):
    """Streaming replaced a spinner that covered the whole turn — rightly, it
    hid three queries and a report behind one unchanging line. But it also
    removed the only feedback before the FIRST token, and a turn that opens
    with an `analyst` call says nothing until that subagent does.

    So the wait indicator is held, and dropped the moment anything reaches the
    screen. Both halves matter: shown at all, and gone before the first print
    rather than left spinning over the answer.
    """
    events: list[str] = []

    class Watching(FakeConsole):
        def status(self, *args, **kwargs):
            events.append("wait:start")

            class _Ctx:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    events.append("wait:stop")
                    return False

            return _Ctx()

        def print(self, *args, **kwargs):
            events.append("print")
            super().print(*args, **kwargs)

    console = Watching()
    answer(console, make_deps(script=["Denim fell 11.8% in Q1."]), "how did denim do?")

    assert "wait:start" in events, "no wait indicator was shown at all"
    assert events.index("wait:stop") < events.index("print"), (
        f"the wait indicator outlived the first output: {events[:6]}"
    )
