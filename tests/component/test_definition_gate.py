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
from retail_agent.agent.deps import TurnContext
from retail_agent.agent.supervisor import build_agent
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.store.definitions import InMemoryDefinitionStore, all_definitions

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


def context_for(user_id: str = "exec", session_id: str = "s1", turn_id: str = "t1") -> TurnContext:
    """The runtime context for a test's identity.

    The tools read identity from `runtime.context`, which `invoke` never
    fills in unless the caller passes it — so both the initial invoke and
    every resume below have to carry it, the same as `cli/chat.py` does.
    Supplied directly rather than read off a `TurnCapture`, which no longer
    carries it.
    """
    return TurnContext(user_id=user_id, session_id=session_id, turn_id=turn_id)


def start(deps, question=QUESTION, arm=True, script=None):
    capture = TurnCapture(question=question)
    agent = build_agent(
        deps, capture, checkpointer=MemorySaver(), pause_for_definitions=arm
    )
    config = {"configurable": {"thread_id": "s1"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
        context=context_for(),
    )
    return agent, config, capture, result


def paused(result) -> bool:
    return bool(result.get("__interrupt__"))


def open_terms(result) -> list[str]:
    """The terms named by the pause itself — our own payload shape now, not
    `HumanInTheLoopMiddleware`'s `action_requests`, and not a side channel on
    the capture (`_approval_gate` is gone; nothing sets `pending_definition`
    any more)."""
    return result["__interrupt__"][0].value["terms"]


def test_asking_pauses_the_turn_before_a_query_is_spent(make_deps):
    """The point of the pause: it happens before the analyst runs, so nothing
    is billed for a question whose meaning is not yet agreed."""
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=asking(), src=source, definitions=InMemoryDefinitionStore())

    _, _, _, result = start(deps)

    assert paused(result)
    assert source.executed == [], "no query before the term is settled"
    assert open_terms(result) == ["LGB"]


def test_the_pause_names_the_term_the_model_could_not_settle(make_deps):
    """A word no list ever contained. This is the case the regex missed."""
    deps = make_deps(script=asking(), definitions=InMemoryDefinitionStore())

    _, _, _, result = start(deps)

    assert "LGB" in open_terms(result)


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

    _, _, _, result = start(deps)

    assert paused(result)
    assert open_terms(result) == ["LGB"]


def test_only_the_unsettled_terms_reach_the_pause(make_deps):
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="exec", term="top", definition="by revenue")
    deps = make_deps(
        script=asking(terms=("top", "LGB")), definitions=definitions
    )

    _, _, _, result = start(deps)

    assert paused(result)
    assert open_terms(result) == ["LGB"]


def test_approving_lets_the_analyst_run_in_the_same_turn(make_deps):
    """The whole point over ending the turn: the answer arrives without the
    executive re-asking their question."""
    definitions = InMemoryDefinitionStore()
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=asking(), src=source, definitions=definitions)

    agent, config, capture, _ = start(deps)
    result = agent.invoke(
        Command(resume={"answers": {"LGB": "low gross basket"}}),
        config,
        context=context_for(),
    )

    assert not paused(result)
    assert source.executed, "the analyst ran once the term was settled"


def test_the_answer_reaches_the_model_that_asked(make_deps):
    """The tool body stores the answer it was resumed with, and uses it right
    there — that is why nothing has to rewrite the tool call's arguments."""
    definitions = InMemoryDefinitionStore()
    deps = make_deps(script=asking(), definitions=definitions)

    agent, config, capture, _ = start(deps)
    agent.invoke(
        Command(resume={"answers": {"LGB": "low gross basket"}}),
        config,
        context=context_for(),
    )

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


def test_the_tool_writes_the_definition_the_executive_gave(make_deps):
    """The CLI must not write before resuming: the tool body replays, and a
    store that changed in between would make `still_open` come back different,
    leaving the `interrupt()` call unreachable and the resume value unconsumed.
    So the answer travels in the resume value and the tool is what stores it."""
    definitions = InMemoryDefinitionStore()
    deps = make_deps(script=asking(), definitions=definitions)

    agent, config, capture, _ = start(deps)
    result = agent.invoke(
        Command(resume={"answers": {"LGB": "low gross basket"}}),
        config,
        context=context_for(),
    )

    assert not paused(result)
    assert all_definitions(definitions, "exec")["lgb"] == "low gross basket"


def test_a_declined_term_is_recorded_as_assumed(make_deps):
    """"You decide" is not an answer to store, and the disclosure is what makes
    it safe: `assumption_note` reads this on the way out."""
    definitions = InMemoryDefinitionStore()
    deps = make_deps(script=asking(), definitions=definitions)

    agent, config, capture, _ = start(deps)
    agent.invoke(
        Command(resume={"answers": {}}), config, context=context_for()
    )

    assert "LGB" in capture.assumed_terms
    assert all_definitions(definitions, "exec") == {}


def test_cancelling_a_later_term_keeps_an_earlier_terms_answer(make_deps):
    """`_settle_definitions` asks about a batch of terms one at a time and
    resumes once. Cancelling on the second must not undo the first: the old
    CLI wrote each answer to the store as it was chosen, so an answer given
    for term one survived a later cancel on term two — it only stopped
    asking, it did not un-decide what was already agreed. All writes are now
    deferred to the tool, so the resume value itself has to carry every
    answer collected so far, not just the ones for terms still open."""
    from .test_repl_turn import FakeConsole
    from retail_agent.cli.chat import _settle_definitions

    definitions = InMemoryDefinitionStore()
    deps = make_deps(
        script=[
            [("ask_for_definitions", {"terms": ["top", "LGB"]})],
            {"definitions": ["the 10 highest by revenue"]},  # propose() for "top"
            {"definitions": ["low gross basket"]},  # propose() for "LGB"
            "All set.",
        ],
        definitions=definitions,
    )

    agent, config, capture, result = start(deps)
    assert paused(result)

    # The executive types an answer for "top", then presses enter on "LGB" —
    # the CLI's cancel path, distinct from the numbered hand-back option.
    console = FakeConsole(["the 10 highest by revenue", ""])
    resume = _settle_definitions(console, deps, capture, {"terms": ["top", "LGB"]})
    result = agent.invoke(Command(resume=resume), config, context=context_for())

    assert not paused(result)
    kept = all_definitions(definitions, "exec")
    assert kept["top"] == "the 10 highest by revenue", (
        "term one's answer must survive the cancel on term two"
    )
    assert "lgb" not in kept
