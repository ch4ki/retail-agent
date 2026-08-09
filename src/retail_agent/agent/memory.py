"""What the agent remembers about the person it is talking to.

Two tools, both write-only from the model's side. `remember_definition` fills a
gap the Golden Bucket does not cover; `note_preference` records, in the user's
own words, how this executive wants answers written.

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
— see `TurnCapture.preference_changes`, which the CLI reports after the answer.
Silent is the failure mode worth avoiding, not immediate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.store.definitions import MAX_DEFINITION_CHARS
from retail_agent.store.preferences import (
    MAX_NOTE_CHARS,
    MAX_NOTES,
    add_note,
    remove_note,
)

log = logging.getLogger(__name__)

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


def build_memory_tools(deps: AgentDeps, capture: TurnCapture) -> list[Callable]:
    """Bound to one turn, because both tools write against its user."""

    def remember_definition(term: str, definition: str) -> str:
        """Record what a business term means for this executive.

        Call this when they tell you — for example "loyal means three or more
        orders in a year". Only for terms whose meaning is a business decision;
        never for a column that already exists.
        """
        with capture.step("remember_definition") as step:
            if deps.definitions is None:
                step.detail = "no store"
                return (
                    "I cannot remember definitions right now, but I will use "
                    "that for this question."
                )

            try:
                deps.definitions.remember(
                    user_id=capture.user_id,
                    term=term.strip().lower(),
                    definition=definition.strip()[:MAX_DEFINITION_CHARS],
                )
            except Exception as err:
                # Not worth failing a turn the user just unblocked.
                log.warning("could not save the definition of %r (%s)", term, err)
                step.detail = f"failed: {err}"
                return "I could not save that, but I will use it for this question."

            step.detail = f"remembered {term}"
            return (
                f"Recorded: {term} means {definition}. I will use that from now "
                f"on. Now answer the original question."
            )

    def note_preference(preference: str, evidence: str) -> str:
        """Record how this executive wants answers written.

        `preference` is the request in plain words — "keep answers under three
        sentences", "show prices in euros", "skip the caveats unless I ask".
        `evidence` must be the exact words they used, quoted from their message.
        Do not call this for questions about the data — "why are sales down"
        says nothing about how they want an answer written.
        """
        with capture.step("note_preference") as step:
            # The check that makes acting on this safe rather than presumptuous.
            # A span the user never typed would mean the model inferred a
            # preference and this tool wrote it down as though they had asked.
            quoted = evidence.strip()
            if not quoted or quoted.lower() not in capture.question.lower():
                step.detail = "evidence not quotable"
                return "Nothing recorded — the evidence must be their exact words."

            if deps.preferences is None:
                step.detail = "no store"
                return "Noted for this answer; I cannot save it as a default."

            note = " ".join(preference.split())
            try:
                outcome = add_note(deps.preferences, user_id=capture.user_id, note=note)
            except Exception as err:
                log.warning("could not save the preference %r (%s)", note, err)
                step.detail = f"failed: {err}"
                return "Noted for this answer; I could not save it as a default."

            if outcome != "added":
                step.detail = f"refused: {outcome}"
                return REFUSALS[outcome]

            # Recorded on the capture, not just described in the reply: the CLI
            # announces the change itself, so the user is told whether or not
            # the model mentions it.
            capture.preference_changes.append(("added", note))
            step.detail = f"added {note}"
            return f"Saved: {note}. I will follow that from now on. Apply it now."

    def forget_preference(preference: str) -> str:
        """Drop a preference this executive no longer wants.

        Pass the saved wording as closely as you can; the match ignores case and
        spacing. Changing a preference is this tool followed by
        `note_preference` with the new wording — there is no edit.
        """
        with capture.step("forget_preference") as step:
            if deps.preferences is None:
                step.detail = "no store"
                return "I cannot change saved preferences right now."

            note = " ".join(preference.split())
            try:
                removed = remove_note(
                    deps.preferences, user_id=capture.user_id, note=note
                )
            except Exception as err:
                log.warning("could not forget the preference %r (%s)", note, err)
                step.detail = f"failed: {err}"
                return "I could not change your saved preferences just now."

            if not removed:
                step.detail = "no match"
                return f"Nothing saved matching {note!r}; nothing changed."

            capture.preference_changes.append(("removed", note))
            step.detail = f"removed {note}"
            return f"Removed: {note}."

    return [remember_definition, note_preference, forget_preference]
