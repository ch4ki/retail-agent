"""Pausing the turn on an unsettled term, rather than ending it.

The gate is the deterministic half of the feature: what stops the turn, and
what is on the far side of the stop. What gets *offered* while it is stopped is
`knowledge/proposals`, and the CLI's part is in `test_repl_turn`.
"""

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.supervisor import build_agent
from retail_agent.store.definitions import InMemoryDefinitionStore

from .conftest import FakeSource

QUESTION = "who are my loyal customers?"


def analysing(question=QUESTION):
    """A supervisor that reaches for the analyst, then reports what it found."""
    return [
        [("analyst", {"question": question})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]


def start(deps, question=QUESTION, arm=True):
    capture = TurnCapture(user_id="exec", session_id="s1", question=question)
    agent = build_agent(
        deps, capture, checkpointer=MemorySaver(), ask_for_definitions=arm
    )
    config = {"configurable": {"thread_id": "s1"}}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    return agent, config, capture, result


def paused(result) -> bool:
    return bool(result.get("__interrupt__"))


def test_an_unsettled_term_pauses_the_turn_before_a_query_is_spent(make_deps):
    """The point of the pause: it happens before the analyst runs, so nothing
    is billed for a question whose meaning is not yet agreed."""
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(
        script=analysing(), src=source, definitions=InMemoryDefinitionStore()
    )

    _, _, capture, result = start(deps)

    assert paused(result)
    assert source.executed == [], "no query before the term is settled"
    assert capture.pending_definition.terms == ("loyal",)


def test_the_pause_names_the_term_and_what_it_turns_on(make_deps):
    deps = make_deps(script=analysing(), definitions=InMemoryDefinitionStore())

    _, _, _, result = start(deps)

    description = result["__interrupt__"][0].value["action_requests"][0]["description"]
    assert "loyal" in description
    assert "what makes a customer loyal" in description


def test_a_definition_in_the_store_settles_it_without_asking(make_deps):
    """Asked once, then reused. Asking the same person the same question every
    week is how a safety feature becomes an irritation."""
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="exec", term="loyal", definition="3 or more orders")
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=analysing(), src=source, definitions=definitions)

    _, _, _, result = start(deps)

    assert not paused(result)
    assert source.executed, "the analyst ran"


def test_approving_lets_the_analyst_run_in_the_same_turn(make_deps):
    """The whole point over today's behaviour: the answer arrives without the
    executive re-asking their question."""
    definitions = InMemoryDefinitionStore()
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=analysing(), src=source, definitions=definitions)

    agent, config, _, _ = start(deps)
    definitions.remember(user_id="exec", term="loyal", definition="3 or more orders")
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config
    )

    assert not paused(result)
    assert source.executed, "the analyst ran once the term was settled"


def test_handing_the_decision_back_is_an_edit_that_sets_assume_undefined(make_deps):
    """"Decide for me" is not a third code path — it is today's assume-and-
    disclose behaviour, reached by rewriting the tool call."""
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(
        script=analysing(), src=source, definitions=InMemoryDefinitionStore()
    )

    agent, config, capture, _ = start(deps)
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "analyst",
                            "args": {"question": QUESTION, "assume_undefined": True},
                        },
                    }
                ]
            }
        ),
        config,
    )

    assert not paused(result)
    assert source.executed, "the analyst ran on its own judgement"
    assert capture.assumed_terms == ["loyal"], "and the assumption is on record"


def test_a_settled_question_is_never_paused(make_deps):
    """A gate that stops ordinary questions is worse than no gate."""
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    question = "how much revenue did we make in March?"
    deps = make_deps(
        script=analysing(question), src=source, definitions=InMemoryDefinitionStore()
    )

    _, _, _, result = start(deps, question)

    assert not paused(result)
    assert source.executed


def test_the_gate_is_off_unless_the_caller_arms_it(make_deps):
    """`ask_once` scores a paused turn as unanswered, and the eval cases with
    undefined terms are the brief's own examples. A headless caller must get
    the analyst's early return, not an interrupt."""
    deps = make_deps(script=analysing(), definitions=InMemoryDefinitionStore())

    _, _, _, result = start(deps, arm=False)

    assert not paused(result)


def test_the_eval_harness_answers_an_undefined_term_rather_than_pausing(make_deps):
    """The regression this whole arming distinction exists to prevent.

    `ask_once` records a paused turn as `[paused awaiting approval]` with no
    rows, and the suite's undefined-term cases are the brief's own examples —
    "why are users in state X underspending", "why did churn spike". An
    always-armed pause would score every one of them as a non-answer, and the
    drop would look like a model regression rather than a wiring change.

    Do not delete this as redundant with the flag test above: that one checks
    `build_agent`, this one checks the caller that would actually suffer.
    """
    from retail_agent.agent.seams import ask_once

    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    headless = [
        # The analyst returns without querying, saying it needs a definition...
        [("analyst", {"question": QUESTION})],
        # ...and the supervisor's prompt tells it to decide and disclose.
        [("analyst", {"question": QUESTION, "assume_undefined": True})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]
    deps = make_deps(
        script=headless, src=source, definitions=InMemoryDefinitionStore()
    )

    answer = ask_once(deps, QUESTION, user="eval")

    assert "paused" not in answer.text
    assert answer.row_count == 1, "it reached a real query and a scorable number"


def test_with_no_definition_store_there_is_nothing_to_ask_into(make_deps):
    """Asking without somewhere to keep the answer means asking again next
    turn, which is worse than assuming and saying so."""
    deps = make_deps(script=analysing(), definitions=None)

    _, _, _, result = start(deps)

    assert not paused(result)
