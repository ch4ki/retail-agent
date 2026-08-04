"""The only file containing control flow.

Every safety property is an edge here, not an instruction in a prompt:
the guard runs before execution, masking runs before synthesis, and the
repair budget is decremented by the graph rather than by the model.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.chat import chat_node
from retail_agent.agent.nodes.execute import execute_node
from retail_agent.agent.nodes.plan import plan_node
from retail_agent.agent.nodes.route import route_node
from retail_agent.agent.nodes.schema_qa import schema_node
from retail_agent.agent.nodes.sql import draft_sql_node
from retail_agent.agent.nodes.synthesize import synthesize_node
from retail_agent.agent.state import TurnState, new_turn_state


def build_graph(deps: AgentDeps, checkpointer=None):
    builder = StateGraph(TurnState)

    builder.add_node("route", partial(route_node, deps=deps))
    builder.add_node("schema", partial(schema_node, deps=deps))
    builder.add_node("chat", partial(chat_node, deps=deps))
    builder.add_node("plan", partial(plan_node, deps=deps))
    builder.add_node("draft_sql", partial(draft_sql_node, deps=deps))
    builder.add_node("execute", partial(execute_node, deps=deps))
    builder.add_node("synthesize", partial(synthesize_node, deps=deps))

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _after_route,
        {"schema": "schema", "chat": "chat", "plan": "plan"},
    )
    builder.add_edge("schema", END)
    builder.add_edge("chat", END)
    builder.add_edge("plan", "draft_sql")

    builder.add_conditional_edges(
        "draft_sql",
        _after_draft,
        {"execute": "execute", "draft_sql": "draft_sql", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "execute",
        _after_execute,
        {"draft_sql": "draft_sql", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)


# --- routing functions: plain Python, no model involved ---


def _after_route(state: TurnState) -> str:
    intent = state.get("intent", "analyze")
    if intent == "schema":
        return "schema"
    if intent == "chat":
        return "chat"
    return "plan"


def _after_draft(state: TurnState) -> str:
    step = _current(state)
    if step is not None and step.sql:
        return "execute"
    if state.get("repair_budget", 0) > 0:
        return "draft_sql"
    return "synthesize"


def _after_execute(state: TurnState) -> str:
    if state.get("step_index", 0) >= len(state.get("plan", [])):
        return "synthesize"
    if _last_attempt_failed(state) and state.get("repair_budget", 0) <= 0:
        return "synthesize"
    return "draft_sql"


def _current(state: TurnState):
    plan = state.get("plan", [])
    index = state.get("step_index", 0)
    return plan[index] if 0 <= index < len(plan) else None


def _last_attempt_failed(state: TurnState) -> bool:
    attempts = state.get("sql_attempts", [])
    return bool(attempts) and attempts[-1].failed


def run_turn(
    graph,
    *,
    user_id: str,
    session_id: str,
    question: str,
    repair_budget: int,
    config: dict | None = None,
) -> TurnState:
    state = new_turn_state(
        user_id=user_id,
        session_id=session_id,
        question=question,
        repair_budget=repair_budget,
    )
    return graph.invoke(state, config=config or {})
