"""Blanks the per-turn scratch space before anything else runs.

This is a node rather than a line inside `route` or `plan` so that the reset is
visible in the graph: the repair budget is a safety property, and every safety
property in this agent is an edge you can point at.
"""

from __future__ import annotations

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.state import TurnState, fresh_scratch


def start_turn_node(state: TurnState, deps: AgentDeps) -> dict:
    return fresh_scratch(
        repair_budget=deps.settings.repair_budget,
        diagnose_budget=deps.settings.diagnose_budget,
    )
