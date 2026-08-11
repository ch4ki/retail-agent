"""What the agent is allowed to remember about the person it is talking to.

The proposal engine this replaced accumulated signals in Postgres and asked at
three sightings, because it sat on top of a *guess* at intent — first a regex,
then a classifier. The honest response to a guess is to propose it rather than
act on it.

This tool is not a guess: it fires only on words the user typed, checked here
against the recorded question. So the rule that must survive is the
quotable-evidence check. Without it the tool writes an inference as though it
were an instruction, which is the failure the proposal engine existed to
prevent.
"""

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.memory import build_memory_tools
from retail_agent.config import Settings
from retail_agent.knowledge.seeds import SEED_TRIOS
from retail_agent.obs.traces import InMemoryTraceStore
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.definitions import InMemoryDefinitionStore
from retail_agent.store.reports import InMemoryReportStore
from retail_agent.store.preferences import InMemoryPreferenceStore


def tools_for(question, trios=(), dense=None):
    deps = AgentDeps(
        settings=Settings(_env_file=None, google_cloud_project="test"),
        llm=object(),
        source=object(),
        policy=PiiPolicy.default(),
        reports=InMemoryReportStore(),
        traces=InMemoryTraceStore(),
        preferences=InMemoryPreferenceStore(),
        definitions=InMemoryDefinitionStore(),
        trios=list(trios),
        dense=dense,
    )
    capture = TurnCapture(user_id="dana", session_id="s1", question=question)
    return {t.name: t.func for t in build_memory_tools(deps, capture)}, deps, capture


def test_a_quoted_preference_is_saved_in_the_users_own_words():
    tools, deps, _ = tools_for("just keep it brief, I don't need the workings")

    tools["note_preference"]("keep answers brief", "keep it brief")

    assert deps.preferences.list_notes(user_id="dana") == ["keep answers brief"]


def test_the_change_is_recorded_so_the_interface_can_announce_it():
    """Applying immediately is defensible; applying *silently* is not. The CLI
    reads this rather than trusting the model to mention what it changed."""
    tools, _, capture = tools_for("keep it brief")

    tools["note_preference"]("keep answers brief", "keep it brief")

    assert capture.preference_changes == [("added", "keep answers brief")]


def test_evidence_the_user_never_typed_changes_nothing():
    """The check that separates "they asked for this" from "the model thinks
    they want this" — and this question is about the data, not the layout.

    This guard is the whole reason the tool may act rather than propose, so it
    outlived the enum validation that used to sit above it."""
    tools, deps, capture = tools_for("why are sales down in Texas?")

    answer = tools["note_preference"]("keep answers brief", "keep it brief")

    assert "exact words" in answer
    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes == []


def test_a_preference_already_saved_is_not_saved_twice():
    tools, deps, capture = tools_for("keep it brief")
    tools["note_preference"]("keep answers brief", "keep it brief")

    answer = tools["note_preference"]("Keep answers brief", "keep it brief")

    assert "already" in answer.lower()
    assert deps.preferences.list_notes(user_id="dana") == ["keep answers brief"]
    assert capture.preference_changes == [("added", "keep answers brief")], "announced once"


def test_a_preference_past_the_cap_is_refused_with_a_way_out():
    from retail_agent.store.preferences import MAX_NOTES

    tools, deps, capture = tools_for("keep it brief")
    deps.preferences.replace_notes(
        user_id="dana", notes=[f"preference {i}" for i in range(MAX_NOTES)]
    )

    answer = tools["note_preference"]("keep answers brief", "keep it brief")

    assert "forget" in answer.lower(), "says how to make room"
    assert len(deps.preferences.list_notes(user_id="dana")) == MAX_NOTES
    assert capture.preference_changes == []


def test_an_over_long_preference_is_refused_rather_than_cut_down():
    """A truncated note is a preference the user did not write."""
    from retail_agent.store.preferences import MAX_NOTE_CHARS

    tools, deps, capture = tools_for("keep it brief")

    answer = tools["note_preference"]("x" * (MAX_NOTE_CHARS + 1), "keep it brief")

    assert "shorter" in answer.lower()
    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes == []


def test_a_store_failure_costs_the_preference_not_the_turn():
    """Losing a preference must not lose the answer it was about."""
    _, deps, capture = tools_for("keep it brief")

    class Broken:
        def list_notes(self, **_):
            raise RuntimeError("postgres is down")

        def replace_notes(self, **_):
            raise RuntimeError("postgres is down")

    object.__setattr__(deps, "preferences", Broken())
    tools = {t.name: t.func for t in build_memory_tools(deps, capture)}

    answer = tools["note_preference"]("keep answers brief", "keep it brief")

    assert "could not save" in answer.lower()
    assert capture.preference_changes == []


def test_forgetting_removes_the_note_and_announces_it():
    tools, deps, capture = tools_for("stop showing prices in euros")
    tools["note_preference"]("show prices in euros", "prices in euros")

    answer = tools["forget_preference"]("show prices in euros")

    assert "removed" in answer.lower()
    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes[-1] == ("removed", "show prices in euros")


def test_forgetting_announces_the_stored_wording_not_the_callers_casing():
    """`remove_note` matches case-insensitively, so the model can ask to forget
    a note using different casing than what was actually saved. What gets
    announced — and eventually shown to the user — has to be what they wrote,
    not the model's paraphrase of its own case."""
    tools, deps, capture = tools_for("stop calling me Manager, my name is Dana")
    deps.preferences.replace_notes(user_id="dana", notes=["Keep answers brief"])

    tools["forget_preference"]("keep answers brief")

    assert deps.preferences.list_notes(user_id="dana") == []
    assert capture.preference_changes == [("removed", "Keep answers brief")]


def test_forgetting_something_that_was_never_saved_says_so():
    tools, deps, capture = tools_for("stop showing prices in euros")

    answer = tools["forget_preference"]("show prices in euros")

    assert "nothing" in answer.lower()
    assert capture.preference_changes == []


def test_editing_a_preference_is_a_forget_then_a_note():
    """There is no edit tool, so this is the path the model has to take —
    and both halves have to reach the announcement."""
    tools, deps, capture = tools_for("make that under two sentences, not three")
    deps.preferences.replace_notes(user_id="dana", notes=["keep answers under three sentences"])

    tools["forget_preference"]("keep answers under three sentences")
    tools["note_preference"]("keep answers under two sentences", "under two sentences")

    assert deps.preferences.list_notes(user_id="dana") == [
        "keep answers under two sentences"
    ]
    assert capture.preference_changes == [
        ("removed", "keep answers under three sentences"),
        ("added", "keep answers under two sentences"),
    ]


def test_a_definition_is_remembered_under_the_term_the_analyst_looks_up():
    """`unresolved` yields lower-cased terms, so a stored 'Loyal' would never be
    found again and the agent would keep asking the same person the same thing."""
    tools, deps, _ = tools_for("loyal means three orders in a year")

    tools["remember_definition"]("Loyal", "three or more orders in a year")

    assert deps.definitions.lookup(user_id="dana", term="loyal") is not None


def test_a_definition_store_failure_costs_the_memory_not_the_turn():
    """The user has just unblocked the agent; failing now wastes that."""
    _, deps, capture = tools_for("loyal means three orders")

    class Broken:
        def remember(self, **_):
            raise RuntimeError("postgres is down")

    object.__setattr__(deps, "definitions", Broken())
    tools = {t.name: t.func for t in build_memory_tools(deps, capture)}

    answer = tools["remember_definition"]("loyal", "three orders")

    assert "could not save" in answer.lower()


# --- asking what a term means ---


def test_asking_reads_back_the_definition_the_executive_just_gave():
    """The pause happens before the tool body runs, and the CLI writes the
    answer to the store. So `approve` needs no rewritten arguments: the tool
    reads back what was just settled."""
    tools, deps, _ = tools_for("report on 10 LGB customers")
    deps.definitions.remember(user_id="dana", term="lgb", definition="low gross basket")

    answer = tools["ask_for_definitions"](["LGB"])

    assert "low gross basket" in answer


def test_with_nobody_to_ask_the_agent_is_told_to_choose_and_disclose():
    """Headless — no interrupt armed, so the body runs against an empty store.
    Refusing here would fail the brief's own eval questions, so the bargain is
    the same one `assumption_note` makes: answer, and say what you assumed."""
    tools, _, capture = tools_for("report on 10 LGB customers")

    answer = tools["ask_for_definitions"](["LGB"])

    assert "LGB" in answer
    assert "state" in answer.lower(), "the disclosure is demanded, not optional"
    assert capture.assumed_terms == ["LGB"], "and the trace records it"


def test_a_term_the_analytics_team_agreed_is_not_treated_as_unsettled():
    """The corpus is consulted before the executive is told nobody defined it.

    Live, this cost five of the eval's forty-seven cases. Asked how many loyal
    customers there are, the `loyal-customers` trio was retrieved — it says
    three or more completed orders, all time — and this tool reported "loyal"
    unsettled anyway, because it only ever read the per-user store. The analyst
    then received the agreed definition *and* an instruction to invent one, and
    answered on a threshold it made up.
    """
    tools, _, capture = tools_for("How many loyal customers do we have?", SEED_TRIOS)

    answer = tools["ask_for_definitions"](["loyal"])

    assert capture.assumed_terms == [], "the corpus settles it; nothing was assumed"
    assert "three or more completed orders" in answer.lower()


def test_the_phrase_the_executive_used_finds_the_term_the_corpus_defines():
    """The model passes the words as they were written, because its own tool
    description tells it to. The corpus keys its definitions on the business
    term alone.

    So "loyal customers" has to find `loyal`. Live, it did not: the answer read
    'There is no agreed definition for "loyal customers"' while the trio
    defining `loyal` sat in the same turn's retrieval.
    """
    tools, _, capture = tools_for("How many loyal customers do we have?", SEED_TRIOS)

    answer = tools["ask_for_definitions"](["loyal customers"])

    assert capture.assumed_terms == []
    assert "three or more completed orders" in answer.lower()


def test_the_executives_own_definition_wins_over_the_corpus():
    """`remember_definition` answers "I will use that from now on". A corpus
    trio arriving later — or retrieved for the first time by a rephrased
    question — must not silently replace the definition this executive was
    promised is in force. The corpus fills gaps; it does not override."""
    tools, deps, _ = tools_for("How many loyal customers do we have?", SEED_TRIOS)
    deps.definitions.remember(
        user_id="dana", term="loyal", definition="five or more orders"
    )

    answer = tools["ask_for_definitions"](["loyal"])

    assert "five or more orders" in answer
    assert "three or more completed orders" not in answer.lower()


def test_a_corpus_settled_term_is_recorded_in_the_trace():
    """The meaning is injected verbatim into the model's context, so the trio
    it came from must show in /trace — an answer that used a corpus definition
    cannot claim it used none."""
    tools, _, capture = tools_for("How many loyal customers do we have?", SEED_TRIOS)

    tools["ask_for_definitions"](["loyal"])

    assert capture.trio_ids, "the consulted trio reaches the trace and the eval"


def test_one_turn_runs_retrieval_once_however_many_places_ask():
    """`ask_for_definitions` calls `settled_meanings` on both sides of its own
    interrupt — once before the pause, again on replay after resume — and with
    dense retrieval configured every run costs an embedding round trip. The
    corpus cannot change mid-turn, so the second call reads the first one's
    cached result."""
    from retail_agent.agent.tools import settled_meanings

    calls = []

    class Dense:
        def rank(self, question, trios):
            calls.append(question)
            return []

    _, deps, capture = tools_for(
        "How many loyal customers do we have?", SEED_TRIOS, dense=Dense()
    )

    settled_meanings(deps, capture)
    settled_meanings(deps, capture)

    assert len(calls) == 1


def test_a_definition_stored_during_the_pause_is_seen_after_it():
    """Only the corpus half of `settled_meanings` may be cached: the personal
    store can change between two calls in the same turn — a sibling
    `remember_definition` call, not a write from inside this pause itself,
    since `interrupt()` replays the whole tool body and a write during the
    pause would make the replayed `still_open` differ and leave the
    `interrupt()` unreachable — and the next caller must see it."""
    from retail_agent.agent.tools import settled_meanings

    _, deps, capture = tools_for("report on 10 LGB customers")

    before = settled_meanings(deps, capture)
    deps.definitions.remember(
        user_id="dana", term="lgb", definition="low gross basket"
    )
    after = settled_meanings(deps, capture)

    assert "lgb" not in before
    assert after["lgb"] == "low gross basket"


def test_partition_splits_the_settled_from_the_still_open():
    """One partition, called by `ask_for_definitions` on both sides of its own
    interrupt — the two calls diverged once and the CLI asked about a term the
    corpus had already settled."""
    from retail_agent.agent.tools import partition_terms

    settled, still_open = partition_terms(
        {"lgb": "low gross basket"}, ["LGB", " top ", "", None]
    )

    assert settled == {"LGB": "low gross basket"}
    assert still_open == ["top"]


def test_a_phrase_is_not_settled_by_a_word_it_merely_contains():
    """The match is on whole words, and only where the defined term is the
    thing being asked about. Without that this degrades into substring
    matching, which would settle "disloyal customers" from `loyal`."""
    tools, _, capture = tools_for("How many disloyal customers?", SEED_TRIOS)

    tools["ask_for_definitions"](["disloyal customers"])

    assert capture.assumed_terms == ["disloyal customers"]


def test_a_term_nobody_agreed_is_still_reported_unsettled():
    """The corpus lookup must not swallow the case it exists to protect: a word
    no trio covers still has to reach the executive."""
    tools, _, capture = tools_for("report on 10 LGB customers", SEED_TRIOS)

    answer = tools["ask_for_definitions"](["LGB"])

    assert capture.assumed_terms == ["LGB"]
    assert "LGB" in answer


def test_a_term_still_open_is_reported_next_to_one_already_settled():
    tools, deps, capture = tools_for("top LGB customers")
    deps.definitions.remember(user_id="dana", term="lgb", definition="low gross basket")

    answer = tools["ask_for_definitions"](["LGB", "top"])

    assert "low gross basket" in answer
    assert capture.assumed_terms == ["top"], "only the unsettled one is assumed"
