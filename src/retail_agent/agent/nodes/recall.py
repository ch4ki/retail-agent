"""Retrieve the analysts' judgement, then find out what is still undecided.

A node rather than a call inside `plan`, for the same reason the confirmation
gate is an edge: it belongs to the shape of the turn. `/trace` then shows which
trios were consulted, which definitions came from the user, and which terms the
agent had to decide alone — exactly what you need when an executive disputes a
number.

Precedence, highest first:

1. **Trios.** A reviewed, versioned decision by the people who own the numbers.
2. **The user's own definitions**, given when the agent asked and nothing in the
   corpus covered it. Fills gaps; never overrides.
3. **Nothing** — the agent asks. If the user declines, it assumes and says so.
"""

from __future__ import annotations

import logging

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.state import TurnState
from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.trios import live_trios, unresolved
from retail_agent.store.definitions import remembered

log = logging.getLogger(__name__)


def recall_node(state: TurnState, deps: AgentDeps) -> dict:
    """Find what settles this question, and what is left over.

    An empty corpus is a valid state, and the important behaviour holds anyway:
    every business term is then unsettled, so the agent asks — and failing an
    answer, says what it assumed rather than quietly choosing.
    """
    question = last_user_message(state)
    if not question:
        return {}

    try:
        found = retrieve(
            question,
            live_trios(deps.trios),
            dense_rank=deps.dense.rank if deps.dense is not None else None,
        )
    except Exception as err:  # retrieval is an improvement, not a dependency
        log.warning("trio retrieval failed (%s); answering without it", err)
        found = []

    open_terms = unresolved(question, found)
    known = remembered(deps.definitions, state.get("user_id", ""), open_terms)

    # Declined this turn: the user pressed enter rather than defining it, so it
    # goes to the assumption path instead of being asked about again.
    declined = set(state.get("declined_terms", []))
    still_open = [t for t in open_terms if t not in known and t not in declined]

    # Only ask if the answer can be kept. Without somewhere to remember it, the
    # agent would ask the same person the same question every turn — which is
    # worse than assuming and saying so.
    if deps.definitions is None:
        return {
            "trio_ids": [trio.id for trio in found],
            "personal_terms": [],
            "assumed_terms": [t for t in open_terms if t not in known],
            "pending_term": "",
        }

    return {
        "trio_ids": [trio.id for trio in found],
        "personal_terms": sorted(known),
        # Anything the user declined is assumed and disclosed.
        "assumed_terms": [t for t in open_terms if t in declined],
        "pending_term": still_open[0] if still_open else "",
    }


def await_definition(state: TurnState) -> dict:
    """A deliberate no-op. The graph breaks *before* this node; the CLI fills
    `definition_reply` in as though this node had produced it."""
    return {}


def apply_definition(state: TurnState, deps: AgentDeps) -> dict:
    """Save what the user said, or record that they declined.

    Either way the term stops being pending, so the loop back through `recall`
    makes progress and cannot ask about the same term twice.
    """
    term = state.get("pending_term", "")
    if not term:
        return {}

    reply = (state.get("definition_reply") or "").strip()
    if not reply:
        # Pressing enter is a valid answer: let the agent choose, and say so.
        return {
            "declined_terms": [*state.get("declined_terms", []), term],
            "pending_term": "",
            "definition_reply": "",
        }

    try:
        deps.definitions.remember(
            user_id=state.get("user_id", ""), term=term, definition=reply
        )
    except Exception as err:  # not worth failing a turn the user just unblocked
        log.warning("could not save the definition of %r (%s)", term, err)

    return {"pending_term": "", "definition_reply": ""}


def recalled(state: TurnState, deps: AgentDeps) -> list:
    """The trio objects for this turn, resolved from the ids in state.

    State holds ids rather than whole trios: a trio carries a full report and a
    query, and a checkpoint per turn should not carry copies of the corpus.
    """
    wanted = set(state.get("trio_ids", []))
    return [trio for trio in live_trios(deps.trios) if trio.id in wanted]


def personal(state: TurnState, deps: AgentDeps) -> dict[str, str]:
    """The user's own definitions in play this turn."""
    return remembered(
        deps.definitions, state.get("user_id", ""), state.get("personal_terms", [])
    )


def definitions_for(state: TurnState, deps: AgentDeps) -> str:
    """Everything settled about this turn's terms: agreed first, then personal.

    Lives here rather than in `sql` because `recall` is what resolves it and
    more than one node downstream needs it. It used to be private to
    `draft_sql`, which is why the planner never received it — and the planner
    decides the shape of the answer.

    Measured live: asked for "engaged customers" the planner emitted a step
    reading "count of distinct users who have placed at least one order", and
    `draft_sql` could not recover, because the word the definition explains was
    already gone from the question it was handed.

    Order matters: where both cover a term the corpus wins, so the model reads
    the reviewed decision before the personal one.
    """
    from retail_agent.knowledge.trios import definitions_block
    from retail_agent.store.definitions import personal_definitions_block

    blocks = [
        definitions_block(recalled(state, deps)),
        personal_definitions_block(personal(state, deps)),
    ]
    return "\n\n".join(block for block in blocks if block)
