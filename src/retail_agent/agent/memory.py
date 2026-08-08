"""What the agent remembers about the person it is talking to.

Two tools, both write-only from the model's side and neither of them able to
change an answer on its own. `remember_definition` fills a gap the Golden Bucket
does not cover; `note_preference` files evidence that a proposal is later built
from. Nothing here is applied silently — a personalisation the reader cannot
account for is worse than none.

`route_node` used to detect a style preference for free, folded into a routing
call the graph was already making. There is no such call now, so detection
became a tool the model elects to call. That is a real loss of guarantee, and
the mitigation is the one rule that made the old detector trustworthy: the
evidence must be a phrase the user actually typed, checked here rather than
believed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.store.definitions import MAX_DEFINITION_CHARS
from retail_agent.store.learning import Signal
from retail_agent.store.preferences import FORMATS

log = logging.getLogger(__name__)

# `standard` is missing from `depth` on purpose: it is the default, and
# proposing a setting someone already has is noise.
STYLE_VALUES: dict[str, frozenset[str]] = {
    "depth": frozenset({"summary", "deep"}),
    "answer_format": frozenset(FORMATS),
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
                return "I cannot remember definitions right now, but I will use that for this question."

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

    def note_preference(field: str, value: str, evidence: str) -> str:
        """Record that the executive said how they want answers presented.

        `field` is 'answer_format' (table, bullets, prose) or 'depth' (summary,
        deep). `evidence` must be the exact words they used, quoted from their
        message. Do not call this for questions about the data — "why are sales
        down" says nothing about presentation.
        """
        with capture.step("note_preference") as step:
            if value not in STYLE_VALUES.get(field, frozenset()):
                step.detail = f"rejected {field}={value}"
                return f"'{value}' is not a value {field} accepts. Nothing recorded."

            # The proposal this evidence eventually produces quotes it back —
            # "you asked for this three times, most recently '<span>'" — and a
            # span the user never typed would make that a fabrication rather
            # than a citation. This is the check that keeps it honest.
            quoted = evidence.strip()
            if not quoted or quoted.lower() not in capture.question.lower():
                step.detail = "evidence not quotable"
                return "Nothing recorded — the evidence must be their exact words."

            if deps.signals is None:
                step.detail = "no store"
                return "Noted for this answer."

            count = deps.signals.record(
                user_id=capture.user_id,
                signal=Signal(field=field, value=value, evidence=quoted),
            )
            step.detail = f"{field}={value} ({count})"
            # Deliberately not applied here. The setting changes only when the
            # user accepts a proposal, so an answer's layout is never something
            # they cannot account for.
            return (
                f"Noted. Apply it to this answer; it becomes their default only "
                f"once they accept the suggestion."
            )

    return [remember_definition, note_preference]
