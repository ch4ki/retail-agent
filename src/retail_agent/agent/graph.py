"""The only file containing control flow.

Every safety property is an edge here, not an instruction in a prompt:
the guard runs before execution, masking runs before synthesis, and the
repair budget is decremented by the graph rather than by the model.
"""

from __future__ import annotations

import time
from functools import partial

from langgraph.graph import END, START, StateGraph

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.chat import chat_node
from retail_agent.agent.nodes.diagnose import diagnose_node
from retail_agent.agent.nodes.execute import execute_node
from retail_agent.agent.nodes.plan import plan_node
from retail_agent.agent.nodes.recall import (
    apply_definition,
    await_definition,
    recall_node,
)
from retail_agent.agent.nodes.report_ops import (
    apply_delete,
    await_confirmation,
    report_ops_node,
)
from retail_agent.agent.nodes.finish_turn import finish_turn_node
from retail_agent.agent.nodes.route import route_node
from retail_agent.agent.nodes.schema_qa import schema_node
from retail_agent.agent.nodes.sql import draft_sql_node
from retail_agent.agent.nodes.start_turn import start_turn_node
from retail_agent.agent.nodes.synthesize import synthesize_node
from retail_agent.agent.state import TurnEvent, TurnState, new_turn_state


def build_graph(deps: AgentDeps, checkpointer=None):
    builder = StateGraph(TurnState)

    def node(name: str, fn):
        builder.add_node(name, _traced(name, fn))

    node("start_turn", partial(start_turn_node, deps=deps))
    node("route", partial(route_node, deps=deps))
    node("schema", partial(schema_node, deps=deps))
    node("chat", partial(chat_node, deps=deps))
    node("recall", partial(recall_node, deps=deps))
    node("await_definition", await_definition)
    node("apply_definition", partial(apply_definition, deps=deps))
    node("plan", partial(plan_node, deps=deps))
    node("report_ops", partial(report_ops_node, deps=deps))
    node("await_confirmation", await_confirmation)
    node("apply_delete", partial(apply_delete, deps=deps))
    node("draft_sql", partial(draft_sql_node, deps=deps))
    node("execute", partial(execute_node, deps=deps))
    node("diagnose", partial(diagnose_node, deps=deps))
    node("synthesize", partial(synthesize_node, deps=deps))
    node("finish_turn", partial(finish_turn_node, deps=deps))

    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "route")
    builder.add_conditional_edges(
        "route",
        _after_route,
        {
            "schema": "schema",
            "chat": "chat",
            "recall": "recall",
            "report_ops": "report_ops",
        },
    )
    builder.add_edge("schema", "finish_turn")
    builder.add_edge("chat", "finish_turn")
    # The destructive path: report_ops reads and stages, the breakpoint pauses,
    # apply_delete writes. Nothing can write before the user answers, because
    # the write is a different node on the far side of the gate.
    builder.add_conditional_edges(
        "report_ops",
        _needs_confirmation,
        {"await_confirmation": "await_confirmation", "end": "finish_turn"},
    )
    builder.add_edge("await_confirmation", "apply_delete")
    builder.add_edge("apply_delete", "finish_turn")
    # Asking what a term means is a breakpoint, like the delete confirmation:
    # declared here rather than inside a node, and the node that acts on the
    # reply sits on the far side of it. The loop back through `recall` is what
    # lets a second undefined term be asked about after the first is settled;
    # `apply_definition` always clears `pending_term`, so it terminates.
    builder.add_conditional_edges(
        "recall",
        _needs_definition,
        {"await_definition": "await_definition", "plan": "plan"},
    )
    builder.add_edge("await_definition", "apply_definition")
    builder.add_edge("apply_definition", "recall")
    builder.add_edge("plan", "draft_sql")

    builder.add_conditional_edges(
        "draft_sql",
        _after_draft,
        {"execute": "execute", "draft_sql": "draft_sql", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "execute",
        _after_execute,
        {
            "draft_sql": "draft_sql",
            "diagnose": "diagnose",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("diagnose", "draft_sql")
    builder.add_edge("synthesize", "finish_turn")
    # Every path leaves through here, which is what makes "the graph records its
    # own turns" true for any caller rather than only for the CLI.
    builder.add_edge("finish_turn", END)

    # The confirmation gate. It lives here, in the file that owns control flow,
    # rather than inside a node — so it is visible in the graph and in Studio's
    # rendering, and so the node before it never re-executes on resume.
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["await_confirmation", "await_definition"],
    )


def _traced(name: str, fn):
    """Wrap a node so it records what it did and how long it took.

    Applied at registration rather than inside each node, so a node added later
    is traced by virtue of being on the graph — it cannot forget to report
    itself. Events accumulate by read-modify-write rather than through a
    reducer, because the graph runs its nodes sequentially and `fresh_scratch`
    has to be able to blank the list at the head of each turn.
    """

    def traced(state: TurnState, *args, **kwargs) -> dict:
        started = time.perf_counter()
        update = fn(state, *args, **kwargs) or {}
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        merged = {**state, **update}
        event = TurnEvent(
            node=name, duration_ms=elapsed_ms, detail=_describe(name, merged)
        )
        # A node that returns `events` wins, so `start_turn` can blank the log
        # for a new turn without its own wrapper putting the old one back.
        prior = update.get("events", state.get("events", []))
        return {**update, "events": [*prior, event]}

    return traced


def _describe(name: str, state: TurnState) -> str:
    """One line of what the node decided, for `/trace`.

    Only values already in state — nothing new is computed and nothing
    unmasked is read, so tracing cannot become its own disclosure path.
    """
    if name == "route":
        return f"intent={state.get('intent', '')}"
    if name == "plan":
        return f"{len(state.get('plan', []))} step(s)"
    if name == "recall":
        ids = state.get("trio_ids", [])
        parts = [f"{len(ids)} trio(s): {', '.join(ids)}" if ids else "no trio matched"]
        if state.get("personal_terms"):
            parts.append(f"user-defined: {', '.join(state['personal_terms'])}")
        if state.get("pending_term"):
            parts.append(f"asking about: {state['pending_term']}")
        if state.get("assumed_terms"):
            parts.append(f"assuming: {', '.join(state['assumed_terms'])}")
        return "; ".join(parts)
    if name == "apply_definition":
        return f"defined {state.get('pending_term') or 'the term'}"
    if name == "diagnose":
        return "empty result — redrafting to match more loosely"
    if name in {"draft_sql", "execute"}:
        attempts = state.get("sql_attempts", [])
        if not attempts:
            return ""
        last = attempts[-1]
        if last.violations:
            return f"{last.step_id}: guard rejected — {'; '.join(last.violations)}"
        if last.error:
            return f"{last.step_id}: {last.error}"
        if last.row_count is not None:
            return f"{last.step_id}: {last.row_count} row(s), {last.bytes_billed} bytes"
        return f"{last.step_id}: drafted, guard passed"
    if name == "report_ops":
        pending = state.get("pending_action")
        return f"staged {len(pending.report_ids)} for deletion" if pending else "answered"
    if name == "apply_delete":
        return f"confirmation={state.get('confirmation', '')!r}"
    if name == "synthesize":
        return f"status={state.get('status', '')}"
    return ""


# --- routing functions: plain Python, no model involved ---


def _after_route(state: TurnState) -> str:
    intent = state.get("intent", "analyze")
    if intent == "schema":
        return "schema"
    if intent == "chat":
        return "chat"
    if intent == "report_op":
        return "report_ops"
    return "recall"


def _needs_definition(state: TurnState) -> str:
    """A term nothing settles means the user is asked before anything runs."""
    return "await_definition" if state.get("pending_term") else "plan"


def _needs_confirmation(state: TurnState) -> str:
    """A staged action means a delete is waiting. Save, list, and a delete that
    matched nothing all answered in the node and stage nothing."""
    return "await_confirmation" if state.get("pending_action") else "end"


def _after_draft(state: TurnState) -> str:
    step = _current(state)
    if step is None:
        # Nothing to draft. draft_sql spends no budget in this case, so
        # routing back to it would cycle until the recursion limit.
        return "synthesize"
    if step.sql:
        return "execute"
    if state.get("repair_budget", 0) > 0:
        return "draft_sql"
    return "synthesize"


def _after_execute(state: TurnState) -> str:
    # A query that succeeded and returned nothing gets one look on its own
    # budget, before the step is treated as answered.
    if _returned_nothing(state) and state.get("diagnose_budget", 0) > 0:
        return "diagnose"
    if state.get("step_index", 0) >= len(state.get("plan", [])):
        return "synthesize"
    if _last_attempt_failed(state) and state.get("repair_budget", 0) <= 0:
        return "synthesize"
    return "draft_sql"


def _returned_nothing(state: TurnState) -> bool:
    """No rows, or the aggregate spelling of no rows.

    `SUM(x) WHERE brand = 'Levis'` against a column holding `Levi's` returns one
    row containing NULL, not zero rows. Checking `row_count == 0` alone misses
    the case this edge exists for — confirmed against live BigQuery.
    """
    attempts = state.get("sql_attempts", [])
    if not attempts:
        return False

    last = attempts[-1]
    if last.failed:
        return False
    if last.row_count == 0:
        return True
    if last.row_count != 1:
        return False

    frame = state.get("frames", {}).get(last.step_id)
    if frame is None or len(frame.rows) != 1:
        return False
    # Every column null in the only row: an aggregate that matched nothing. A
    # row with any value in it is a real answer, even a partial one.
    return all(value is None for value in frame.rows[0])


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
    config: dict | None = None,
) -> TurnState:
    state = new_turn_state(
        user_id=user_id, session_id=session_id, question=question
    )
    return graph.invoke(state, config=config or {})
