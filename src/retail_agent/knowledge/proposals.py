"""Candidate definitions for a term nobody has settled.

The agent already refuses to invent a definition silently: `unresolved` finds
the term, and the analyst stops before spending a query on it. What was missing
was a way to *answer* that question without ending the turn. This produces the
options the CLI offers.

The options are a convenience, not the mechanism. The user is asked either way,
and can always type their own or hand the decision back. So every failure here
returns an empty list — the same bargain `recall` makes about retrieval, for
the same reason: a question the agent is right to ask must not depend on a
model call succeeding.

Deliberately not part of the interrupt predicate. What decides whether to stop
the turn stays deterministic; only what is *offered* once it has stopped comes
from a model.
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
here — specifically, {hint}.

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
    hint: str,
    schema: str,
    settled: dict[str, str] | None = None,
) -> list[str]:
    """Definitions to offer for `term`. Never raises, never blocks the prompt."""
    prompt = PROMPT.format(
        question=question,
        term=term,
        hint=hint,
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
