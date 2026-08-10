"""Pausing the turn on an unsettled term, rather than ending it.

The gate used to be the deterministic half of the feature: a regex over
nineteen hardcoded words decided what stopped the turn. It could only pause on
a term somebody had thought of in advance, so "10 LGB customers" sailed
through and came back with a confident number.

Now the model asks, by calling `ask_for_definitions`, and the tool call is what
stops the turn. What is left deterministic is the part worth keeping: a term
this executive has already defined is filtered out before anything pauses, so
the promise that they are asked once still holds.

What gets *offered* while it is stopped is `knowledge/proposals`, and the CLI's
part is in `test_repl_turn`.
"""

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.supervisor import build_agent
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.store.definitions import InMemoryDefinitionStore

from .conftest import FakeSource

QUESTION = "who are my LGB customers?"


def asking(question=QUESTION, terms=("LGB",)):
    """A supervisor that asks what the term means, then analyses."""
    return [
        [("ask_for_definitions", {"terms": list(terms)})],
        [("analyst", {"question": question})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]


def straight_to_analysis(question=QUESTION):
    """A supervisor that understood the question and never asked."""
    return [
        [("analyst", {"question": question})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]


def start(deps, question=QUESTION, arm=True, script=None):
    capture = TurnCapture(user_id="exec", session_id="s1", question=question)
    agent = build_agent(
        deps, capture, checkpointer=MemorySaver(), pause_for_definitions=arm
    )
    config = {"configurable": {"thread_id": "s1"}}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    return agent, config, capture, result


def paused(result) -> bool:
    return bool(result.get("__interrupt__"))


def test_asking_pauses_the_turn_before_a_query_is_spent(make_deps):
    """The point of the pause: it happens before the analyst runs, so nothing
    is billed for a question whose meaning is not yet agreed."""
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=asking(), src=source, definitions=InMemoryDefinitionStore())

    _, _, capture, result = start(deps)

    assert paused(result)
    assert source.executed == [], "no query before the term is settled"
    assert capture.pending_definition.terms == ("LGB",)


def test_the_pause_names_the_term_the_model_could_not_settle(make_deps):
    """A word no list ever contained. This is the case the regex missed."""
    deps = make_deps(script=asking(), definitions=InMemoryDefinitionStore())

    _, _, _, result = start(deps)

    description = result["__interrupt__"][0].value["action_requests"][0]["description"]
    assert "LGB" in description


def test_a_term_the_executive_already_defined_is_not_asked_again(make_deps):
    """Asked once, then reused. This is the one thing left that does not
    depend on the model behaving: it is filtered before the pause."""
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="exec", term="LGB", definition="low gross basket")
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=asking(), src=source, definitions=definitions)

    _, _, _, result = start(deps)

    assert not paused(result)
    assert source.executed, "the analyst ran"


def test_a_term_the_corpus_defines_does_not_pause_the_turn(make_deps):
    """The gate reads the agreed corpus, not only this executive's own answers.

    Reported from a live session: asked "how many loyal customers do we have?",
    the CLI stopped and offered four invented meanings of "loyal", while the
    `loyal-customers` trio — three or more completed orders, all time — was in
    the corpus the same turn would retrieve.

    The tool body already consulted the corpus. This gate runs *before* the
    body, so arming the interrupt was enough to bypass that entirely: headless
    callers got the agreed definition and the CLI did not.
    """
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(
        script=asking(question="how many loyal customers do we have?", terms=("loyal customers",)),
        src=source,
        definitions=InMemoryDefinitionStore(),
        trios=list(SEED_TRIOS),
    )

    _, _, _, result = start(deps, question="how many loyal customers do we have?")

    assert not paused(result), "the analytics team already defined 'loyal'"
    assert source.executed, "the analyst ran"


def test_a_term_no_trio_covers_still_pauses(make_deps):
    """The corpus lookup must not swallow the case the gate exists for."""
    deps = make_deps(
        script=asking(), definitions=InMemoryDefinitionStore(), trios=list(SEED_TRIOS)
    )

    _, _, capture, result = start(deps)

    assert paused(result)
    assert capture.pending_definition.terms == ("LGB",)


def test_only_the_unsettled_terms_reach_the_pause(make_deps):
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="exec", term="top", definition="by revenue")
    deps = make_deps(
        script=asking(terms=("top", "LGB")), definitions=definitions
    )

    _, _, capture, result = start(deps)

    assert paused(result)
    assert capture.pending_definition.terms == ("LGB",)


def test_approving_lets_the_analyst_run_in_the_same_turn(make_deps):
    """The whole point over ending the turn: the answer arrives without the
    executive re-asking their question."""
    definitions = InMemoryDefinitionStore()
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=asking(), src=source, definitions=definitions)

    agent, config, _, _ = start(deps)
    definitions.remember(user_id="exec", term="LGB", definition="low gross basket")
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert not paused(result)
    assert source.executed, "the analyst ran once the term was settled"


def test_the_answer_reaches_the_model_that_asked(make_deps):
    """`approve` runs the tool body, which reads the store back. That is why
    nothing has to rewrite the tool call's arguments."""
    definitions = InMemoryDefinitionStore()
    deps = make_deps(script=asking(), definitions=definitions)

    agent, config, _, _ = start(deps)
    definitions.remember(user_id="exec", term="LGB", definition="low gross basket")
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert any("low gross basket" in prompt for prompt in deps.llm.prompts)


def test_a_question_the_model_understood_is_never_paused(make_deps):
    """A gate that stops ordinary questions is worse than no gate. Nothing
    pauses unless the model says it needs to ask."""
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(
        script=straight_to_analysis(), src=source, definitions=InMemoryDefinitionStore()
    )

    _, _, _, result = start(deps)

    assert not paused(result)
    assert source.executed


def test_the_gate_is_off_unless_the_caller_arms_it(make_deps):
    """`ask_once` scores a paused turn as unanswered, and the eval cases with
    undefined terms are the brief's own examples. A headless caller must get an
    answer with a disclosure, not an interrupt."""
    deps = make_deps(script=asking(), definitions=InMemoryDefinitionStore())

    _, _, capture, result = start(deps, arm=False)

    assert not paused(result)
    assert capture.assumed_terms == ["LGB"], "assumed rather than asked"


def test_the_eval_harness_answers_an_undefined_term_rather_than_pausing(make_deps):
    """The regression this arming distinction exists to prevent.

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
    deps = make_deps(
        script=asking(), src=source, definitions=InMemoryDefinitionStore()
    )

    answer = ask_once(deps, QUESTION, user="eval")

    assert "paused" not in answer.text
    assert answer.row_count == 1, "it reached a real query and a scorable number"


def test_with_no_definition_store_there_is_nothing_to_ask_into(make_deps):
    """Asking without somewhere to keep the answer means asking again next
    turn, which is worse than assuming and saying so."""
    deps = make_deps(script=asking(), definitions=None)

    _, _, _, result = start(deps)

    assert not paused(result)
