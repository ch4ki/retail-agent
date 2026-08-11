"""The REPL side of the pause: what is offered, and what is done with the answer.

The gate itself is in `test_definition_gate`. These are the tests about the
executive's experience of it — that the answer arrives in the same turn they
asked, and that what was shown is what got recorded.
"""

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver

from retail_agent.cli.chat import _answer
from retail_agent.store.definitions import InMemoryDefinitionStore

from .conftest import FakeSource
from .test_repl_turn import FakeConsole

QUESTION = "who are my loyal customers?"

OPTIONS = {
    "definitions": [
        "3 or more completed orders, ever",
        "2 or more orders in the last 12 months",
        "ordered at least once in the last 90 days",
    ]
}


def script(question=QUESTION, options=OPTIONS, terms=("loyal",)):
    """Supervisor asks what the term means; the pause interrupts; then the query."""
    return [
        [("ask_for_definitions", {"terms": list(terms)})],
        options,  # consumed by `propose` while the turn is paused
        [("analyst", {"question": question})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]


def run(make_deps, typed, turns=None, definitions=None):
    definitions = definitions if definitions is not None else InMemoryDefinitionStore()
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(
        script=turns or script(), src=source, definitions=definitions
    )
    console = FakeConsole(typed)
    trace = _answer(console, deps, MemorySaver(), "dana", "s1", QUESTION)
    return console, deps, definitions, source, trace


def test_the_options_are_offered_under_the_term(make_deps):
    """No gloss on what the term turns on: that used to be a dict value keyed
    by the word, and the words are the executive's own now. The options say
    what the gloss said, concretely."""
    console, *_ = run(make_deps, ["1"])

    printed = console.text()
    assert "loyal" in printed
    assert "3 or more completed orders, ever" in printed
    assert "2 or more orders in the last 12 months" in printed


def test_choosing_an_option_records_it_and_answers_in_the_same_turn(make_deps):
    """The whole point over ending the turn: the executive does not re-ask."""
    console, _, definitions, source, _ = run(make_deps, ["2"])

    kept = definitions.lookup(user_id="dana", term="loyal")
    assert kept.definition == "2 or more orders in the last 12 months"
    assert source.executed, "the analyst ran once the term was settled"
    assert "Nine of them." in console.text()


def test_typing_your_own_definition_is_the_one_recorded(make_deps):
    """The generated options are a convenience; they are not the only way in."""
    console, _, definitions, source, _ = run(
        make_deps, ["someone who ordered in three different months"]
    )

    kept = definitions.lookup(user_id="dana", term="loyal")
    assert kept.definition == "someone who ordered in three different months"
    assert source.executed


def test_the_something_else_option_prompts_for_the_text(make_deps):
    """Picking the numbered escape hatch and then typing is the same outcome as
    typing straight away — the number is a signpost, not a second path."""
    _, _, definitions, source, _ = run(make_deps, ["4", "ordered in three months"])

    kept = definitions.lookup(user_id="dana", term="loyal")
    assert kept.definition == "ordered in three months"
    assert source.executed


def test_handing_it_back_records_nothing_and_the_agent_says_what_it_assumed(make_deps):
    """"Decide for me" must not write a definition the executive never chose."""
    console, _, definitions, source, trace = run(make_deps, ["5"])

    assert definitions.lookup(user_id="dana", term="loyal") is None
    assert source.executed, "the analyst still answered"
    assert trace.assumptions == ["loyal"], "and the trace says it assumed"


def test_an_empty_answer_cancels_without_querying(make_deps):
    """Someone who does not want to settle it now must not be billed for a
    query against a meaning nobody agreed."""
    turns = [
        [("ask_for_definitions", {"terms": ["loyal"]})],
        OPTIONS,
        "I need a definition of loyal before I can answer that.",
    ]
    console, deps, definitions, source, _ = run(make_deps, [""], turns=turns)

    assert "3 or more completed orders, ever" in console.text(), "it did ask"
    assert source.executed == []
    assert definitions.lookup(user_id="dana", term="loyal") is None
    # The CLI no longer crafts its own reject message — a cancel resumes with
    # `{"answers": {}}`, same shape as a hand-back, and it is the tool's own
    # `NOBODY_TO_ASK` that tells the model why the analyst did not run.
    assert "Nobody is available to settle: loyal" in deps.llm.prompts[-1], (
        "the model has to be told why the analyst did not run"
    )


def test_a_term_already_defined_is_never_asked_about_again(make_deps):
    """Asked once, then reused."""
    definitions = InMemoryDefinitionStore()
    definitions.remember(user_id="dana", term="loyal", definition="3 or more orders")
    turns = [
        [("ask_for_definitions", {"terms": ["loyal"]})],
        [("analyst", {"question": QUESTION})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]
    console, _, _, source, _ = run(make_deps, [], turns=turns, definitions=definitions)

    assert "3 or more completed orders" not in console.text(), "no prompt"
    assert source.executed


def test_two_open_terms_are_asked_about_one_at_a_time_in_order(make_deps):
    """Each prompt stays a simple choice, and the second is generated knowing
    how the first was settled."""
    question = "who are my top loyal customers?"
    turns = [
        [("ask_for_definitions", {"terms": ["top", "loyal"]})],
        {"definitions": ["the 10 highest by revenue"]},  # top
        {"definitions": ["3 or more orders, ever"]},  # loyal
        [("analyst", {"question": question})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]
    definitions = InMemoryDefinitionStore()
    source = FakeSource(frames={"default": pd.DataFrame({"id": [1]})})
    deps = make_deps(script=turns, src=source, definitions=definitions)
    console = FakeConsole(["1", "1"])

    _answer(console, deps, MemorySaver(), "dana", "s1", question)

    assert definitions.lookup(user_id="dana", term="top").definition == (
        "the 10 highest by revenue"
    )
    assert definitions.lookup(user_id="dana", term="loyal").definition == (
        "3 or more orders, ever"
    )
    assert source.executed, "and then it answered"

    settling_loyal = deps.llm.prompts[2]
    assert "the 10 highest by revenue" in settling_loyal, (
        "the second prompt should read the first answer"
    )


def test_offering_options_does_not_probe_the_warehouse_for_column_values(make_deps):
    """The options are plain English, not SQL, so the values a column holds buy
    nothing here. The analyst pays for that scan because it stops it writing
    `gender = 'female'` against a column holding 'F'; a menu cannot make that
    mistake, and a pause should not cost a warehouse round trip per term."""

    class Probing(FakeSource):
        def column_values(self, table, columns):
            self.probes.append(table)
            return {}

    class Watching(FakeConsole):
        """Snapshots the probe count at the moment the question is asked.

        Measured here rather than after the turn: the analyst scans values for
        good reason once it is running, so the count afterwards says nothing
        about what the prompt itself cost.
        """

        def input(self, prompt=""):
            self.probes_when_asked = len(source.probes)
            return super().input(prompt)

    source = Probing(frames={"default": pd.DataFrame({"id": [1]})})
    source.probes = []
    deps = make_deps(
        script=script(), src=source, definitions=InMemoryDefinitionStore()
    )
    console = Watching(["1"])

    _answer(console, deps, MemorySaver(), "dana", "s1", QUESTION)

    assert console.probes_when_asked == 0, "settling a term must not scan values"
    assert source.probes, "the analyst still scans them when it writes SQL"


def test_losing_the_options_still_leaves_a_question_that_can_be_answered(make_deps):
    """A model failure costs the menu, not the pause. `propose` returns nothing
    and the executive types their own."""
    turns = [
        [("ask_for_definitions", {"terms": ["loyal"]})],
        {"wrong_field": []},  # the schema rejects it; propose returns []
        [("analyst", {"question": QUESTION})],
        [("run_sql", {"sql": "SELECT id FROM users"})],
        "Nine of them.",
        "Nine of them.",
    ]
    console, _, definitions, source, _ = run(
        make_deps, ["two orders in a year"], turns=turns
    )

    assert definitions.lookup(user_id="dana", term="loyal").definition == (
        "two orders in a year"
    )
    assert source.executed
