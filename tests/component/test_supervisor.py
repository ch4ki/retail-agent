"""One turn, end to end, through the compiled agent.

These are the tests about the stack rather than about a tool: what reaches the
model, what never runs, and what is left behind for `/trace` afterwards.
"""

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.prompts import SAFETY_RULES
from retail_agent.agent.subagents import final_text
from retail_agent.agent.supervisor import build_agent
from retail_agent.store.personas import InMemoryPersonaStore
from retail_agent.store.preferences import InMemoryPreferenceStore, add_note

from .conftest import FakeSource


def run(deps, question, user="exec"):
    capture = TurnCapture(user_id=user, session_id="s1", question=question)
    agent = build_agent(deps, capture, checkpointer=MemorySaver())
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": "s1"}},
    )
    return final_text(result), capture


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

    stored = traces.get(owner_id="exec", turn_id=capture.turn_id)
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

    stored = traces.get(owner_id="exec", turn_id=capture.turn_id)
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
    from retail_agent.agent.capture import TurnCapture
    from retail_agent.agent.schema import build_schema_tool

    deps = make_deps(src=source)
    capture = TurnCapture()
    describe = build_schema_tool(deps, capture)[0]

    rendered = describe()

    assert source.executed == [], "answering what data exists must cost nothing"
    assert "order_items" in rendered
    assert capture.events[0][2] == "4 table(s)"


def test_a_turn_runs_through_the_provider_chain(make_deps):
    """The agent is compiled over `ResilientChatModel`, not over a raw model.

    This is the test that would have caught `bind_tools` being missing, and it
    is also the evidence for there being no `__getattr__` on the chain: a whole
    turn needs `bind_tools` and `invoke` and nothing else.
    """
    from retail_agent.llm.resilience import ResilientChatModel

    deps = make_deps(script=["Hello."])
    chain = ResilientChatModel([("primary", deps.llm), ("fallback", deps.llm)])
    object.__setattr__(deps, "llm", chain)

    answer, _ = run(deps, "hello")

    assert answer == "Hello."
