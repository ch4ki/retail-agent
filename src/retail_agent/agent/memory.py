"""What the agent asks the person it is talking to, and what it keeps.

`ask_for_definitions` is the one tool here that reads rather than writes, and
the only one that can stop a turn. It replaced a regex over nineteen hardcoded
words, which could only ever pause on a term somebody had thought of in advance
— see the LGB case in
`docs/superpowers/specs/2026-08-09-model-driven-term-detection-design.md`.
The model now decides what it does not understand, which is the one judgement it
is better placed to make than a list is.

The rest are write-only from the model's side. `remember_definition` keeps what
this executive answers, so the question is asked once; `note_preference` records,
in the user's own words, how they want answers written.

There used to be a proposal engine behind the second one: signals accumulated in
Postgres, and at three sightings the agent asked whether to make it the default.
That machinery answered a question this design no longer asks. It existed
because *inference* might be wrong — a regex, and later a classifier, guessing
at intent from phrasing — and the honest response to a guess is to propose it
rather than act on it.

The tool is not a guess. It fires only when the model has quoted words the user
actually typed, checked here against the recorded question, so "apply it" and
"ask whether to apply it" collapse into the same thing: the user already said
it. What replaced the proposal is the requirement that the change is *announced*
— see `TurnState["preference_changes"]`, which the CLI reports after the
answer. Silent is the failure mode worth avoiding, not immediate.
"""

from __future__ import annotations

import logging
import time

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command, interrupt

from retail_agent.agent.deps import AgentDeps, TurnContext
from retail_agent.agent.state import step_event
from retail_agent.agent.tools import partition_terms, settled_meanings
from retail_agent.llm.messages import message_text
from retail_agent.store.definitions import MAX_DEFINITION_CHARS
from retail_agent.store.preferences import (
    MAX_NOTE_CHARS,
    MAX_NOTES,
    add_note,
    remove_note,
)

log = logging.getLogger(__name__)

# What `ask_for_definitions` says about a term that is still unsettled once
# the pause is resumed — whether the person gave no answer for it, declined
# outright, or simply never got asked because there was nobody there. All of
# those resume the same interrupt() call and land here the same way; nothing
# distinguishes "the executive said decide for yourself" from "the executive
# typed nothing", and nothing needs to.
#
# Not a refusal. The brief's own example questions turn on undefined terms, and
# an agent that declines to answer them is not safe, it is useless. Same bargain
# `assumption_note` makes on the way out: decide, and disclose the decision.
NOBODY_TO_ASK = (
    "Nobody is available to settle: {terms}. Do not ask again and do not "
    "refuse. Choose one concrete, defensible rule for each — a threshold, a "
    "window or a ranking — use it, and state the rule you applied in your "
    "answer."
)

# What to say when the store accepted the call but not the note. Keyed by what
# `add_note` returns, so a new outcome there is a KeyError here rather than a
# tool that reports success for something it did not save.
REFUSALS = {
    "empty": "Nothing recorded — there was no preference in that.",
    "duplicate": "That is already saved; nothing changed.",
    "too_long": (
        f"That is over {MAX_NOTE_CHARS} characters, so I did not save it. "
        f"Say it shorter and I will."
    ),
    "full": (
        f"You already have {MAX_NOTES} preferences saved. Ask me to forget one "
        f"first, then tell me this again."
    ),
}


def build_memory_tools(
    deps: AgentDeps, *, pause_for_definitions: bool = False
) -> list[BaseTool]:
    """Bound to one turn, because both tools write against its user.

    `_recall_cache` is `settled_meanings`' memoisation cell, created fresh
    here rather than carried on anything longer-lived: every real caller
    rebuilds the agent per turn, so a plain dict closed over here lives
    exactly one turn too, without needing a name for itself in checkpointed
    state. See `settled_meanings` for why the cache exists at all.
    """
    _recall_cache: dict = {}

    @tool
    def remember_definition(
        term: str, definition: str, runtime: ToolRuntime[TurnContext, object]
    ) -> Command:
        """Record what a business term means for this executive.

        Call this when they tell you — for example "loyal means three or more
        orders in a year". Only for terms whose meaning is a business decision;
        never for a column that already exists.
        """
        started = time.perf_counter()
        if deps.definitions is None:
            detail = "no store"
            return _reply(
                runtime,
                "I cannot remember definitions right now, but I will use "
                "that for this question.",
                "remember_definition",
                started,
                detail,
            )

        try:
            deps.definitions.remember(
                user_id=runtime.context.user_id,
                term=term.strip().lower(),
                definition=definition.strip()[:MAX_DEFINITION_CHARS],
            )
        except Exception as err:
            # Not worth failing a turn the user just unblocked.
            log.warning("could not save the definition of %r (%s)", term, err)
            detail = f"failed: {err}"
            return _reply(
                runtime,
                "I could not save that, but I will use it for this question.",
                "remember_definition",
                started,
                detail,
            )

        detail = f"remembered {term}"
        return _reply(
            runtime,
            f"Recorded: {term} means {definition}. I will use that from now "
            f"on. Now answer the original question.",
            "remember_definition",
            started,
            detail,
        )

    @tool
    def ask_for_definitions(
        terms: list[str], runtime: ToolRuntime[TurnContext, object]
    ) -> Command:
        """Ask the executive what a business term means, before querying.

        Use this when the question turns on a word whose meaning is a business
        decision rather than a column — an in-house label, a segment name, a
        threshold, a ranking — and neither the agreed definitions nor this
        executive's own cover it. Pass the words exactly as they wrote them.

        Call it before `analyst`, not after. A query written against a guessed
        definition has already spent the money and produced a number nobody can
        trace to a decision.
        """
        started = time.perf_counter()
        question = _last_human_text(runtime.state)
        # The same lookup and the same partition the CLI's gate used to do,
        # now in the one place that needs the answer.
        known = settled_meanings(
            deps, question, user_id=runtime.context.user_id, cache=_recall_cache
        )
        settled, still_open = partition_terms(known, terms)
        # `settled_meanings` caches the corpus retrieval in `_recall_cache`, so
        # this reads what was just consulted rather than paying for a second
        # lookup.
        consulted = [trio.id for trio in (_recall_cache.get("trios") or [])]

        # Only pause if the answer can be kept: without a store the agent
        # would ask the same person the same question every turn, which is
        # worse than assuming and saying so.
        if still_open and pause_for_definitions and deps.definitions is not None:
            reply = interrupt(
                {"kind": "ask_for_definitions", "terms": list(still_open)}
            )
            answers = reply.get("answers") or {}
            for term, meaning in answers.items():
                try:
                    deps.definitions.remember(
                        user_id=runtime.context.user_id,
                        term=term.strip().lower(),
                        definition=meaning.strip()[:MAX_DEFINITION_CHARS],
                    )
                except Exception as err:
                    # Not worth failing a turn the executive just unblocked.
                    log.warning(
                        "could not save the definition of %r (%s)", term, err
                    )
            settled.update(answers)
            still_open = [term for term in still_open if term not in answers]

        detail = _describe_settled(settled, still_open)

        parts = []
        if settled:
            lines = "\n".join(f"- {t}: {d}" for t, d in settled.items())
            parts.append(
                f"These are already defined. Use them exactly:\n{lines}"
            )
        if still_open:
            parts.append(NOBODY_TO_ASK.format(terms=", ".join(still_open)))
        answer = "\n\n".join(parts) or "There was nothing to settle."

        return _reply(
            runtime,
            answer,
            "ask_for_definitions",
            started,
            detail,
            assumed_terms=list(still_open),
            trio_ids=consulted,
        )

    @tool
    def note_preference(
        preference: str, evidence: str, runtime: ToolRuntime[TurnContext, object]
    ) -> Command:
        """Record how this executive wants answers written.

        `preference` is the request in plain words — "keep answers under three
        sentences", "show prices in euros", "skip the caveats unless I ask".
        `evidence` must be the exact words they used, quoted from their message.
        Do not call this for questions about the data — "why are sales down"
        says nothing about how they want an answer written.
        """
        started = time.perf_counter()
        # The check that makes acting on this safe rather than presumptuous.
        # A span the user never typed would mean the model inferred a
        # preference and this tool wrote it down as though they had asked.
        # Read off the turn's own messages, which is where identity and the
        # transcript both live now.
        question = _last_human_text(runtime.state)
        quoted = evidence.strip()
        if not quoted or quoted.lower() not in question.lower():
            detail = "evidence not quotable"
            return _reply(
                runtime,
                "Nothing recorded — the evidence must be their exact words.",
                "note_preference",
                started,
                detail,
            )

        if deps.preferences is None:
            detail = "no store"
            return _reply(
                runtime,
                "Noted for this answer; I cannot save it as a default.",
                "note_preference",
                started,
                detail,
            )

        note = " ".join(preference.split())
        try:
            outcome = add_note(
                deps.preferences, user_id=runtime.context.user_id, note=note
            )
        except Exception as err:
            log.warning("could not save the preference %r (%s)", note, err)
            detail = f"failed: {err}"
            return _reply(
                runtime,
                "Noted for this answer; I could not save it as a default.",
                "note_preference",
                started,
                detail,
            )

        if outcome != "added":
            detail = f"refused: {outcome}"
            return _reply(
                runtime, REFUSALS[outcome], "note_preference", started, detail
            )

        detail = f"added {note}"
        return _reply(
            runtime,
            f"Saved: {note}. I will follow that from now on. Apply it now.",
            "note_preference",
            started,
            detail,
            preference_changes=[{"action": "added", "note": note}],
        )

    @tool
    def forget_preference(
        preference: str, runtime: ToolRuntime[TurnContext, object]
    ) -> Command:
        """Drop a preference this executive no longer wants.

        Pass the saved wording as closely as you can; the match ignores case and
        spacing. Changing a preference is this tool followed by
        `note_preference` with the new wording — there is no edit.
        """
        started = time.perf_counter()
        if deps.preferences is None:
            detail = "no store"
            return _reply(
                runtime,
                "I cannot change saved preferences right now.",
                "forget_preference",
                started,
                detail,
            )

        note = " ".join(preference.split())
        try:
            # Found before it is removed, so what gets announced is the
            # stored wording rather than the caller's casing of it —
            # `remove_note` matches case-insensitively, so the two can
            # differ, and the CLI must quote back what the user actually
            # saved, not the model's paraphrase of its case.
            stored = next(
                (
                    existing
                    for existing in deps.preferences.list_notes(
                        user_id=runtime.context.user_id
                    )
                    if " ".join(existing.split()).lower() == note.lower()
                ),
                None,
            )
            if stored is None:
                detail = "no match"
                return _reply(
                    runtime,
                    f"Nothing saved matching {note!r}; nothing changed.",
                    "forget_preference",
                    started,
                    detail,
                )

            remove_note(
                deps.preferences, user_id=runtime.context.user_id, note=stored
            )
        except Exception as err:
            log.warning("could not forget the preference %r (%s)", note, err)
            detail = f"failed: {err}"
            return _reply(
                runtime,
                "I could not change your saved preferences just now.",
                "forget_preference",
                started,
                detail,
            )

        detail = f"removed {stored}"
        return _reply(
            runtime,
            f"Removed: {stored}.",
            "forget_preference",
            started,
            detail,
            preference_changes=[{"action": "removed", "note": stored}],
        )

    return [
        ask_for_definitions,
        remember_definition,
        note_preference,
        forget_preference,
    ]


def _describe_settled(settled: dict[str, str], still_open: list[str]) -> str:
    parts = []
    if settled:
        parts.append(f"settled: {', '.join(settled)}")
    if still_open:
        parts.append(f"unanswered: {', '.join(still_open)}")
    return "; ".join(parts) or "nothing asked"


def _last_human_text(state: object) -> str:
    """The question this turn is actually about, read off the messages.

    Identity and the transcript both live in graph state, so this reads it
    from `runtime.state` directly — the last `HumanMessage` in the list is
    this turn's question, whatever came before it in the conversation.
    """
    messages = (state or {}).get("messages", []) or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def _reply(
    runtime: ToolRuntime,
    text: str,
    name: str,
    started: float,
    detail: str,
    **extra: object,
) -> Command:
    """Every memory tool's answer, wrapped for `TurnState`.

    One `ToolMessage` plus this call's own `events` entry and `calls: 1` —
    the shape every path through every tool here ends on — plus whatever else
    this particular call contributed (`preference_changes`, `assumed_terms`,
    `trio_ids`, ...), passed as keyword updates.
    """
    update = {
        "messages": [ToolMessage(content=text, tool_call_id=runtime.tool_call_id)],
        "events": [step_event(name, started, detail)],
        "calls": 1,
    }
    update.update(extra)
    return Command(update=update)
