"""Retrieve the analysts' judgement before planning anything.

A node rather than a call inside `plan`, for the same reason the confirmation
gate is an edge: it belongs to the shape of the turn. `/trace` then shows which
trios were consulted and which terms the agent had to decide for itself, which
is exactly what you need when an executive disputes a number.
"""

from __future__ import annotations

import logging

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.state import TurnState
from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.trios import unresolved

log = logging.getLogger(__name__)


def recall_node(state: TurnState, deps: AgentDeps) -> dict:
    """Find the trios that bear on this question, and note what none of them
    settle.

    An empty corpus is a valid state, and the important behaviour holds anyway:
    every business term is then unresolved, so the agent says what it assumed
    rather than quietly choosing.
    """
    question = last_user_message(state)
    if not question:
        return {}

    try:
        found = retrieve(question, deps.trios)
    except Exception as err:  # retrieval is an improvement, not a dependency
        log.warning("trio retrieval failed (%s); answering without it", err)
        found = []

    return {
        "trio_ids": [trio.id for trio in found],
        "assumed_terms": unresolved(question, found),
    }


def recalled(state: TurnState, deps: AgentDeps) -> list:
    """The trio objects for this turn, resolved from the ids in state.

    State holds ids rather than whole trios: a trio carries a full report and a
    query, and a checkpoint per turn should not carry copies of the corpus.
    """
    wanted = set(state.get("trio_ids", []))
    return [trio for trio in deps.trios if trio.id in wanted]
