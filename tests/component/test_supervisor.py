"""One turn, end to end, through the compiled agent.

These are the tests about the stack rather than about a tool: what reaches the
model, what never runs, and what is left behind for `/trace` afterwards.
"""

import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import TurnContext
from retail_agent.agent.prompts import SAFETY_RULES
from retail_agent.agent.subagents import final_text
from retail_agent.agent.supervisor import build_agent
from retail_agent.store.personas import InMemoryPersonaStore
from retail_agent.store.preferences import InMemoryPreferenceStore, add_note

from .conftest import FakeSource


# The fixed id every `run()` call mints its `TurnContext` with — a test's own
# stand-in for the caller doing the minting, now that `TurnCapture` no longer
# does. Constant rather than random: each test's `traces` store is fresh, so
# there is nothing for two turns sharing it to collide on.
TURN_ID = "t1"


def run(deps, question, user="exec"):
    capture = TurnCapture(question=question)
    agent = build_agent(deps, capture, checkpointer=MemorySaver())
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": "s1"}},
        context=TurnContext(user_id=user, session_id="s1", turn_id=TURN_ID),
    )
    return final_text(result), capture


def test_a_turn_with_no_context_fails_loudly(make_deps):
    """The regression this guards: on the LangGraph server a run posted with
    no `context` is coerced to `TurnContext()`, not left `None` — so without
    this check every identity-scoped tool would silently read and write
    against the empty-string user rather than fail. Checked here with no
    context passed at all, which — in a direct in-process `invoke` — leaves
    `runtime.context` `None`, the other shape this guard has to catch."""
    from retail_agent.agent.middleware import MissingTurnIdentity

    deps = make_deps(script=["Hello."])
    capture = TurnCapture(question="hello")
    agent = build_agent(deps, capture, checkpointer=MemorySaver())

    with pytest.raises(MissingTurnIdentity):
        agent.invoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"configurable": {"thread_id": "s1"}},
        )


def test_a_turn_with_an_empty_user_id_fails_the_same_way(make_deps):
    """The server-coerced shape: a `TurnContext` that exists but carries no
    `user_id`, which `_coerce_context` produces from a request body that sent
    `context: {}` — this must fail exactly like no context at all."""
    from retail_agent.agent.middleware import MissingTurnIdentity

    deps = make_deps(script=["Hello."])
    capture = TurnCapture(question="hello")
    agent = build_agent(deps, capture, checkpointer=MemorySaver())

    with pytest.raises(MissingTurnIdentity):
        agent.invoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            {"configurable": {"thread_id": "s1"}},
            context=TurnContext(user_id="", session_id="s1"),
        )


def test_a_dict_context_is_coerced_into_a_real_turncontext(make_deps):
    """The shape the LangGraph server actually sends: a run's `context` is
    parsed from the JSON request body, so it arrives as a plain `dict`, never
    a `TurnContext` instance. LangGraph only turns that dict into the
    dataclass because `build_agent` declares `context_schema=TurnContext` —
    nothing pins that line. Delete it and `_coerce_context(None, {...})`
    returns the dict unchanged (verified directly against the installed
    `langgraph` package), so every tool's `runtime.context.user_id` becomes
    an `AttributeError` on its first identity-scoped call: a dict has no such
    attribute. Every other test in this suite passes a `TurnContext`
    instance to `context=`, so this is the only one that would notice.

    `list_reports` is the probe: it reads `runtime.context.user_id` and
    nothing else, so the turn completing at all — through a real tool call,
    not just agent construction — is the proof the coercion ran.
    """
    deps = make_deps(script=[[("list_reports", {})], "You have no saved reports yet."])
    capture = TurnCapture(question="what have I saved")
    agent = build_agent(deps, capture, checkpointer=MemorySaver())

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what have I saved"}]},
        {"configurable": {"thread_id": "s1"}},
        context={"user_id": "exec", "session_id": "s1", "turn_id": "t1"},
    )

    assert final_text(result) == "You have no saved reports yet."
    assert capture.events and capture.events[0][0] == "list_reports"


def test_a_question_answerable_from_the_conversation_needs_no_tool(make_deps):
    """A greeting is answered directly, without reaching for the analyst."""
    deps = make_deps(script=["Hello — ask me about orders or revenue."])

    answer, _ = run(deps, "hello, what can you do?")

    assert "orders" in answer


def test_the_persona_and_the_safety_rules_reach_the_model(make_deps):
    """The persona is a row the CEO edits weekly, read per model call."""
    personas = InMemoryPersonaStore()
    personas.save(name="terse", body="Answer in at most two sentences.", updated_by="ceo")
    personas.activate(name="terse")
    deps = make_deps(script=["Hi."], personas=personas)

    run(deps, "hello")

    prompt = deps.llm.prompts[0]
    assert "at most two sentences" in prompt
    assert SAFETY_RULES.splitlines()[1] in prompt


def test_a_preference_change_lands_without_a_restart(make_deps):
    """Bound per model call, not when the agent was built."""
    prefs = InMemoryPreferenceStore()
    add_note(prefs, user_id="exec", note="answer in bullet points")
    deps = make_deps(script=["Hi."], preferences=prefs)

    run(deps, "hello")

    assert "bullet points" in deps.llm.prompts[0]


def test_every_turn_leaves_a_trace(make_deps, traces):
    """Recorded by middleware on every path out, so it holds for any caller —
    the CLI, the eval harness and Studio alike."""
    source = FakeSource(frames={"default": pd.DataFrame({"revenue": [12]})})
    deps = make_deps(
        script=[
            [("analyst", {"question": "what was revenue?"})],
            "Revenue was 12.",
        ],
        src=source,
    )
    # The analyst subagent shares the model, so its turns queue behind these.
    deps.llm.script[1:1] = [
        [("run_sql", {"sql": "SELECT SUM(sale_price) AS revenue FROM order_items"})],
        "12.",
    ]

    answer, capture = run(deps, "what was revenue?")

    stored = traces.get(owner_id="exec", turn_id=TURN_ID)
    assert stored is not None
    assert stored.intent == "analyze"
    assert stored.question == "what was revenue?"
    # Innermost first: `capture.step` files on exit, so the query the analyst
    # ran is recorded before the analyst call that contained it.
    assert [name for name, _, _ in stored.events] == ["run_sql", "analyst"]
    assert stored.attempts and stored.attempts[0]["row_count"] == 1


def test_a_trace_carries_no_row_values(make_deps, traces):
    """A trace must not become a second disclosure path."""
    source = FakeSource(
        frames={"default": pd.DataFrame({"id": [1], "email": ["a@b.com"]})}
    )
    deps = make_deps(
        script=[
            [("analyst", {"question": "list customers"})],
            [("run_sql", {"sql": "SELECT id FROM users"})],
            "One customer.",
            "One customer.",
        ],
        src=source,
    )

    _, capture = run(deps, "list customers")

    stored = traces.get(owner_id="exec", turn_id=TURN_ID)
    assert "a@b.com" not in str(stored)


def test_the_final_answer_is_not_swept_for_contact_details(make_deps):
    """The output sweep is gone: PII is stopped where it enters, not where it leaves.

    Asserted rather than deleted, because this test used to claim the opposite
    and a reader deserves to see which way the decision went. The three inbound
    layers — SQL guard, `mask_dataframe`, `_pii()` — mean the model is never
    shown personal data, so the only thing a final-answer regex could still
    match is something the model made up.
    """
    deps = make_deps(script=["Contact them at dana@example.com."])

    answer, capture = run(deps, "who should I call?")

    assert "dana@example.com" in answer
    assert capture.status == "ok"


def test_describe_schema_costs_no_query_and_says_what_it_found(make_deps, source):
    """The trace line is the only place a reader learns this path is free.

    It read "0 table(s)" against a live warehouse holding six, because the count
    grepped for a string the DDL renderer does not emit.
    """
    from langchain.tools import ToolRuntime

    from retail_agent.agent.capture import TurnCapture
    from retail_agent.agent.schema import build_schema_tool

    deps = make_deps(src=source)
    capture = TurnCapture()
    describe = build_schema_tool(deps, capture)[0].func
    runtime = ToolRuntime(
        state=None,
        context=TurnContext(),
        config={},
        stream_writer=None,
        tool_call_id="test",
        store=None,
    )

    rendered = describe(runtime=runtime)

    assert source.executed == [], "answering what data exists must cost nothing"
    assert "order_items" in rendered
    assert capture.events[0][2] == "4 table(s)"


class DeadProvider(BaseChatModel):
    """A provider that is down, and counts how often it was asked."""

    message: str = "503 Service Unavailable"

    def __init__(self, message: str = "503 Service Unavailable", **kwargs):
        super().__init__(message=message, **kwargs)
        object.__setattr__(self, "calls", [])

    @property
    def _llm_type(self) -> str:
        return "dead"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        raise RuntimeError(self.message)


def with_dead_primary(deps, message="503 Service Unavailable", attempts=3):
    """Put a dead provider in front, and the working double behind it."""
    dead = DeadProvider(message)
    object.__setattr__(deps, "llm_fallbacks", [deps.llm])
    object.__setattr__(deps, "llm", dead)
    object.__setattr__(
        deps, "settings", deps.settings.model_copy(update={"llm_retry_attempts": attempts})
    )
    return dead


class FlakyProvider(DeadProvider):
    """Down for the first call, up for the rest."""

    answer: str = "Hello."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise RuntimeError(self.message)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


def test_a_transient_failure_is_retried_on_the_same_provider(make_deps):
    """The single-provider deployment, which is most of them.

    No fallbacks here on purpose: this is the case that had no protection at
    all when retry lived in a wrapper that `build_llm` only applied to chains
    of more than one. A transient `Connection error.` ended the turn, and one
    blip cost 37 of 47 cases in a live eval run.

    This is the only test that actually waits out a backoff — one delay of
    `BASE_DELAY_SECONDS`, jittered.
    """
    deps = make_deps(script=[])
    flaky = FlakyProvider()
    object.__setattr__(deps, "llm", flaky)

    answer, _ = run(deps, "hello")

    assert answer == "Hello."
    assert len(flaky.calls) == 2, "retried on the same provider, not abandoned"
    assert deps.llm_fallbacks == [], "and with nothing to fall back to"


def test_a_dead_primary_falls_over_to_the_fallback(make_deps):
    """The brief's resilience claim, end to end through a compiled agent.

    `attempts=1` so the turn does not actually sit through the backoff; that
    the backoff is configured at all is asserted in `test_resilience.py`.
    """
    deps = make_deps(script=["Hello."])
    dead = with_dead_primary(deps, attempts=1)

    answer, _ = run(deps, "hello")

    assert answer == "Hello.", "the fallback answered"
    assert len(dead.calls) == 1


def test_a_rejected_key_reaches_the_fallback_without_retrying(make_deps):
    """`is_retryable` wired into the middleware, observed rather than asserted.

    Retries are left at their configured 3 here: a 401 is permanent, so the
    turn should move on after one attempt. If the classifier were not passed
    through, this would take the full budget — and the two backoff waits are
    what makes that visible as a slow test rather than a silent one.
    """
    deps = make_deps(script=["Hello."])
    dead = with_dead_primary(deps, message="401 invalid api key")

    answer, _ = run(deps, "hello")

    assert answer == "Hello."
    assert len(dead.calls) == 1, "a rejected key is not worth a second attempt"


@pytest.mark.asyncio
async def test_an_async_turn_falls_over_too(make_deps):
    """The same turn, driven the way LangGraph Server drives it.

    `create_agent` compiles its model node as a sync/async pair and LangGraph
    picks the half that matches how the graph was entered. Every other test in
    this file — like the CLI and the evals — calls `invoke`, so the async half
    was never run offline. That is how the chain object that used to sit here
    shipped without an `ainvoke`: an `AttributeError` on the first model call
    of every Studio turn, with a green suite behind it.

    Middleware carries both halves itself, which is the point of the swap.
    """
    deps = make_deps(script=["Hello."])
    dead = with_dead_primary(deps, attempts=1)

    capture = TurnCapture(question="hello")
    agent = build_agent(deps, capture, checkpointer=MemorySaver())
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        {"configurable": {"thread_id": "s1"}},
        context=TurnContext(user_id="exec", session_id="s1", turn_id="t1"),
    )

    assert final_text(result) == "Hello."
    assert len(dead.calls) == 1
