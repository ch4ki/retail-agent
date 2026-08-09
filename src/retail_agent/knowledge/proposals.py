"""Candidate definitions for a term nobody has settled.

The agent already refuses to invent a definition silently: it calls
`ask_for_definitions`, and the interrupt stops the turn before a query is spent
on a guess. What was missing was a way to *answer* that question without ending
the turn. This produces the options the CLI offers.

The options are a convenience, not the mechanism. The user is asked either way,
and can always type their own or hand the decision back. So every failure here
returns an empty list — the same bargain `recall` makes about retrieval, for
the same reason: a question the agent is right to ask must not depend on a
model call succeeding.

This module used to end by saying the interrupt predicate stayed deterministic
— that only what is *offered* once the turn has stopped came from a model. That
held while a regex over nineteen hardcoded words decided what stopped it, and
it is what let "10 LGB customers" through: a word nobody had thought to add did
not exist. The model now decides, by calling `ask_for_definitions`, and this
module generates the options for whatever it names. What is left deterministic
is narrower and better chosen — whether the executive has already answered.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from retail_agent.store.definitions import MAX_DEFINITION_CHARS

log = logging.getLogger(__name__)

# Past four, a numbered list stops being a choice and becomes a wall.
MAX_OPTIONS = 4

PROMPT = """\
An executive asked: {question}

Their question turns on the term "{term}", and nobody has agreed what it means
here. Work out for yourself what has to be decided about it — a threshold, a
window, a ranking, a boundary — and then propose the answers.

Propose {count} different, concrete definitions they could pick from. Each one:

- is a rule a person could check: a threshold, a time window, or a ranking
- is written in plain English, not SQL, and fits on one line
- can actually be computed from the tables below
- is meaningfully different from the others, not the same rule reworded

Order them with the most conventional reading first.

Tables available:
{schema}
{settled}"""

SETTLED = """
Already agreed with this executive during this question — stay consistent with
these rather than contradicting them:
{lines}"""


class Proposals(BaseModel):
    """What the model returns. A list of plain-English rules."""

    definitions: list[str] = Field(
        description="Candidate meanings for the term, most conventional first."
    )


def propose(
    llm,
    *,
    question: str,
    term: str,
    schema: str,
    settled: dict[str, str] | None = None,
) -> list[str]:
    """Definitions to offer for `term`. Never raises, never blocks the prompt."""
    prompt = PROMPT.format(
        question=question,
        term=term,
        count=MAX_OPTIONS,
        schema=schema,
        settled=_settled_block(settled or {}),
    )

    try:
        reply = llm.with_structured_output(Proposals).invoke(prompt)
    except Exception as err:
        # Includes the schema rejecting the reply, which is the same outcome
        # from the user's side: no options, and the two fixed choices remain.
        log.warning("could not propose definitions for %r (%s)", term, err)
        return []

    return _tidy(getattr(reply, "definitions", None) or [])


def _settled_block(settled: dict[str, str]) -> str:
    if not settled:
        return ""
    lines = "\n".join(f"- {term}: {meaning}" for term, meaning in settled.items())
    return SETTLED.format(lines=lines)


def _tidy(definitions: list[str]) -> list[str]:
    """Order-preserving, deduplicated, and cut to what the store will keep.

    The truncation is not cosmetic: offering an option longer than `remember`
    will save would show the user one definition and record a different one.
    """
    kept: list[str] = []
    for definition in definitions:
        cleaned = " ".join(str(definition).split())[:MAX_DEFINITION_CHARS]
        if cleaned and cleaned not in kept:
            kept.append(cleaned)
    return kept[:MAX_OPTIONS]
