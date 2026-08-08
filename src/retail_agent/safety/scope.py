"""The input guard: what the agent refuses to engage with at all.

Deliberately narrow, and worth being precise about what it is and is not.

It is **not** a topic classifier. Deciding whether a question is "about the
data" needs a model call on every turn, and the graph's router — the one call
that could have carried it for free — is gone. Scope in the ordinary sense is
held by the tools instead: every tool this agent has reads retail data or the
user's own report library, so a question about the weather has nothing to
answer it with, and the system prompt says to decline politely.

What this catches is the adversarial case, where a model's willingness to be
helpful is the attack surface: attempts to override the instructions, to
extract the prompt, to have PII read out, or to have the agent write to the
warehouse. Those are lexical and worth stopping before a model sees them,
because the cost of a false negative is not a bad answer.
"""

from __future__ import annotations

import re

# Each entry is (why, pattern). The reason is returned to the user, so it has to
# be true and specific — "that looks like an attempt to override my
# instructions" is checkable; "I can't help with that" is not.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "that reads as an attempt to override the instructions I run under",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b"
            r"(previous|prior|above|earlier|all|your)\b[^.]{0,20}"
            r"\b(instruction|rule|prompt|direction|guideline)",
            re.I,
        ),
    ),
    (
        "I do not show my own configuration",
        re.compile(
            r"\b(reveal|show|print|repeat|output|tell me|what is|what's)\b[^.]{0,30}"
            r"\b(your|the)\b[^.]{0,20}\b(system prompt|initial prompt|instructions)\b",
            re.I,
        ),
    ),
    (
        "customer contact details are masked and cannot be shown, in aggregate "
        "or individually",
        re.compile(
            r"\b(email|e-mail|phone|address|postcode|zip|coordinates|latitude|"
            r"longitude|full name)\b[^.]{0,40}\b(of|for|list|show|give|reveal|"
            r"unmask|export|dump)\b"
            # Plurals spelled out rather than left to `\b`: "phone numbers"
            # does not match `\bphone number\b`, and that near-miss is exactly
            # the phrasing someone would use.
            r"|\b(list|show|give|reveal|unmask|export|dump)\b[^.]{0,40}"
            r"\b(emails?|e-mails?|phone numbers?|home address(?:es)?|"
            r"street address(?:es)?|postal address(?:es)?|coordinates)\b",
            re.I,
        ),
    ),
    (
        "the warehouse connection is read-only, so nothing can be written to it",
        re.compile(
            r"\b(drop|truncate|alter)\s+table\b"
            r"|\bdelete\s+from\b|\binsert\s+into\b|\bupdate\s+\w+\s+set\b"
            r"|\bgrant\s+(all|select|insert)\b",
            re.I,
        ),
    ),
)


def refuse(question: str) -> str | None:
    """The reason to refuse, or None to let the turn run.

    First match wins; the rules do not overlap in practice and ranking them
    would imply a precedence that is not real.
    """
    for reason, pattern in RULES:
        if pattern.search(question):
            return reason
    return None
