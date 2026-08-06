"""The Golden Bucket: how analysts here decided what a question means.

The brief's example questions cannot be answered from the schema alone. "Why
are users in state X underspending?" — underspending relative to what? "Why did
churn spike?" — theLook has no subscriptions, no cancellations; churn is not
merely undefined, it cannot be read off the columns at all. Someone decided it
means "ordered before, nothing in the trailing 90 days".

Without that, the agent still answers. It picks a definition, writes clean SQL,
and returns a confident number that does not match what Finance reports, with
nothing in the output showing a guess was made. That silent failure is what
this package exists to prevent.

A trio holds more than three strings. `metric_definitions` is the field that
carries the value: injecting a past *query* copies old date filters and joins
into a question they do not fit, while injecting the *definition* carries the
analyst's judgement and lets the agent write fresh SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

# Words that look like columns and are not. Each one is a business judgement
# somebody has to have made; the agent must never quietly make it alone.
#
# Deliberately conservative. A term here that is actually unambiguous costs an
# unnecessary caveat; a term missing from here costs a confident wrong number.
UNDEFINED_TERMS: dict[str, str] = {
    "churn": "which customers count as churned, and over what window",
    "churned": "which customers count as churned, and over what window",
    "underspending": "what spend level counts as low, and compared to whom",
    "overspending": "what spend level counts as high, and compared to whom",
    "top": "how many, and ranked by which measure",
    "best": "ranked by which measure",
    "worst": "ranked by which measure",
    "healthy": "what threshold makes a figure healthy",
    "unhealthy": "what threshold makes a figure unhealthy",
    "performing well": "what counts as performing well",
    "underperforming": "what counts as underperforming",
    "at risk": "what puts a customer at risk, and over what window",
    "loyal": "what makes a customer loyal",
    "engaged": "what makes a customer engaged",
    "active": "what recency makes a customer active",
    "inactive": "what recency makes a customer inactive",
    "high value": "what makes a customer high value",
    "recently": "how many days counts as recent",
    "lately": "how many days counts as recent",
}

_TERM_PATTERNS = tuple(
    (term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
    for term in UNDEFINED_TERMS
)


@dataclass(frozen=True)
class Trio:
    """Question → SQL → report, plus the judgement that connects them."""

    id: str
    question: str
    sql: str
    report: str
    # The part that actually gets reused. {"churn": "ordered in the prior 180
    # days and nothing in the trailing 90, excluding cancelled and returned"}
    metric_definitions: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    author: str = "analyst"
    approved_at: datetime | None = None
    version: int = 1
    # Definitions change. Superseding rather than editing means a report from
    # last quarter can still be read against the definition that produced it.
    superseded_by: str | None = None

    def defines(self, term: str) -> bool:
        return term.lower() in {k.lower() for k in self.metric_definitions}


def undefined_terms(question: str) -> list[str]:
    """Business terms in the question that the schema cannot settle.

    Longer phrases win over the words inside them, so "performing well" is one
    finding rather than also matching nothing else. Order follows the question,
    so the caveat reads in the order the user wrote.
    """
    found: list[tuple[int, str]] = []
    for term, pattern in _TERM_PATTERNS:
        match = pattern.search(question)
        if match:
            found.append((match.start(), term))

    found.sort()
    kept: list[str] = []
    for _, term in found:
        # "top" inside "performing well" would be nonsense, but "at risk"
        # contains "risk" and "high value" contains "value" — drop any term
        # that is a substring of one already kept.
        if any(term in other and term != other for other in kept):
            continue
        kept = [k for k in kept if not (k in term and k != term)]
        kept.append(term)
    return kept


def unresolved(question: str, trios: list[Trio]) -> list[str]:
    """Terms the question raises that no retrieved trio defines.

    This is what the graph branches on. A term with a definition behind it is
    answered from the analyst's judgement; a term without one has to be
    surfaced, because the alternative is inventing it silently.
    """
    return [
        term
        for term in undefined_terms(question)
        if not any(trio.defines(term) for trio in trios)
    ]


def definitions_block(trios: list[Trio]) -> str:
    """The definitions to put in front of the model, deduplicated.

    Only definitions — not the past SQL. A previous query carries its own date
    filters and joins, and pasting it into a new question is how an agent
    answers last quarter's question with this quarter's label.
    """
    seen: dict[str, str] = {}
    for trio in trios:
        for term, meaning in trio.metric_definitions.items():
            seen.setdefault(term.lower(), meaning)

    if not seen:
        return ""

    lines = "\n".join(f"- {term}: {meaning}" for term, meaning in sorted(seen.items()))
    return (
        "Definitions agreed by the analytics team. Use these exactly; do not "
        f"substitute your own:\n{lines}"
    )


def style_examples(trios: list[Trio], limit: int = 2) -> str:
    """How analysts here actually write a finding.

    The `report` field is hard to specify and easy to demonstrate: split by
    cohort, compare against a baseline, close with numbered actions.
    """
    reports = [trio.report.strip() for trio in trios if trio.report.strip()][:limit]
    if not reports:
        return ""
    joined = "\n\n---\n\n".join(reports)
    return f"How analysts here write findings. Match this shape, not this content:\n\n{joined}"


def assumption_note(terms: list[str]) -> str:
    """What the agent says when it had to decide something itself.

    Stated in the answer rather than logged, because the number is only
    trustworthy if the reader knows which judgement produced it.
    """
    if not terms:
        return ""
    described = "; ".join(f"**{term}** — {UNDEFINED_TERMS[term]}" for term in terms)
    return (
        "State plainly, before the answer, that no agreed definition exists for "
        f"the following and say which one you are using: {described}. "
        "One short sentence. Do not apologise, and do not refuse to answer."
    )
