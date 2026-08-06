"""Learning a user's preferences from how they ask.

The brief asks the agent to learn over time — how deep someone likes the
analysis, whether they want tables or prose. Two rules shape this:

**Proposed, never applied silently.** Personalisation that changes the output
without saying why is worse than none, because the reader cannot tell whether
the agent changed or the data did. Evidence accumulates; the agent asks.

**The evidence is quotable.** Detection is deterministic phrase matching rather
than a model judgement, so the proposal can say *you asked for this three times,
most recently "just give me the numbers"*. "The model inferred you prefer
brevity" is not something a user can check or argue with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Evidence needed before the agent asks. Two is a coincidence.
PROPOSAL_THRESHOLD = 3

# After a decline, the same suggestion needs a lot more evidence — being asked
# twice about something you already refused is how a helpful feature becomes an
# irritating one.
DECLINED_MULTIPLIER = 3


@dataclass(frozen=True)
class Signal:
    field: str
    value: str
    evidence: str  # the phrase the user actually used


@dataclass(frozen=True)
class Proposal:
    field: str
    value: str
    count: int
    evidence: str

    def question(self) -> str:
        return (
            f"You've asked for this {self.count} times — most recently "
            f'"{self.evidence}". Set {self.field} to {self.value}? '
            f"[bold]/prefs accept[/bold] or [bold]/prefs decline[/bold]"
        )


# Phrases that ask for less. Deliberately explicit: each one is something a
# person types when they want a shorter answer, not merely a short question.
_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("depth", "summary", r"\bjust (?:give me )?(?:the )?(?:numbers?|figures?|totals?)\b"),
    ("depth", "summary", r"\b(?:keep it |be )?(?:brief|short|concise)\b"),
    ("depth", "summary", r"\b(?:one|single) (?:line|sentence)\b"),
    ("depth", "summary", r"\bno (?:detail|explanation|preamble)\b"),
    ("depth", "summary", r"\btl;?dr\b"),
    # Asking to go further, which is not the same as asking a causal question.
    ("depth", "deep", r"\b(?:more|further) detail\b"),
    ("depth", "deep", r"\bgo deeper\b"),
    ("depth", "deep", r"\b(?:tell me more|elaborate|expand on)\b"),
    ("depth", "deep", r"\bin depth\b"),
    ("depth", "deep", r"\bbreak (?:it|that) down\b"),
    ("answer_format", "table", r"\b(?:as|in) a table\b"),
    ("answer_format", "table", r"\btabulate\b"),
    ("answer_format", "bullets", r"\bbullets?\b"),
    ("answer_format", "bullets", r"\b(?:as|in) a list\b"),
    ("answer_format", "prose", r"\b(?:as|in) (?:prose|a paragraph)\b"),
    ("answer_format", "prose", r"\bwrite it out\b"),
)

_COMPILED = tuple(
    (field, value, re.compile(pattern, re.IGNORECASE))
    for field, value, pattern in _PATTERNS
)


def detect(message: str) -> list[Signal]:
    """Preference signals in one user message.

    Note what is *not* here: "why", "explain", "how come". Those are the most
    common analysis questions in the brief — "why are users in state X
    underspending?" is a request about the data, not a request for a longer
    answer. Treating them as depth signals would mean the agent slowly decided
    every analyst wanted essays.
    """
    found: list[Signal] = []
    seen: set[tuple[str, str]] = set()
    for field, value, pattern in _COMPILED:
        match = pattern.search(message)
        if match and (field, value) not in seen:
            seen.add((field, value))
            found.append(Signal(field=field, value=value, evidence=match.group(0)))
    return found


@runtime_checkable
class SignalStore(Protocol):
    def record(self, *, user_id: str, signal: Signal) -> int: ...

    def counts(self, *, user_id: str) -> dict[tuple[str, str], tuple[int, str]]: ...

    def decline(self, *, user_id: str, field: str, value: str) -> None: ...

    def declines(self, *, user_id: str) -> dict[tuple[str, str], int]: ...

    def clear(self, *, user_id: str, field: str) -> None: ...


class InMemorySignalStore:
    def __init__(self) -> None:
        self._counts: dict[str, dict[tuple[str, str], tuple[int, str]]] = {}
        self._declines: dict[str, dict[tuple[str, str], int]] = {}

    def record(self, *, user_id: str, signal: Signal) -> int:
        mine = self._counts.setdefault(user_id, {})
        key = (signal.field, signal.value)
        count = mine.get(key, (0, ""))[0] + 1
        mine[key] = (count, signal.evidence)
        return count

    def counts(self, *, user_id: str) -> dict[tuple[str, str], tuple[int, str]]:
        return dict(self._counts.get(user_id, {}))

    def decline(self, *, user_id: str, field: str, value: str) -> None:
        mine = self._declines.setdefault(user_id, {})
        mine[(field, value)] = mine.get((field, value), 0) + 1

    def declines(self, *, user_id: str) -> dict[tuple[str, str], int]:
        return dict(self._declines.get(user_id, {}))

    def clear(self, *, user_id: str, field: str) -> None:
        """Once a field is settled, its evidence stops mattering — otherwise
        accepting a proposal leaves the counters that produced it in place."""
        mine = self._counts.get(user_id, {})
        for key in [k for k in mine if k[0] == field]:
            del mine[key]


def next_proposal(
    signals: SignalStore,
    *,
    user_id: str,
    current,
    threshold: int = PROPOSAL_THRESHOLD,
) -> Proposal | None:
    """The strongest suggestion worth making, or nothing.

    Nothing is proposed for a setting the user already has, and a suggestion
    they declined needs several times the evidence before it comes back.
    """
    declines = signals.declines(user_id=user_id)
    best: Proposal | None = None

    for (field, value), (count, evidence) in signals.counts(user_id=user_id).items():
        if getattr(current, field, None) == value:
            continue
        needed = threshold * (DECLINED_MULTIPLIER ** declines.get((field, value), 0))
        if count < needed:
            continue
        if best is None or count > best.count:
            best = Proposal(field=field, value=value, count=count, evidence=evidence)

    return best
