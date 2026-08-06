"""Drafts SQL and puts it through the guard before anything can execute it."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.recall import recalled
from retail_agent.agent.nodes.schema_qa import render_schema
from retail_agent.agent.prompts import REPAIR_PROMPT, SQL_PROMPT
from retail_agent.agent.state import AnalysisStep, SqlAttempt, TurnState
from retail_agent.llm.messages import message_text
from retail_agent.knowledge.trios import definitions_block, sql_assumption_note
from retail_agent.safety.sql_guard import check_sql

FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def current_step(state: TurnState) -> AnalysisStep | None:
    """Return the plan step the graph is currently working on, or None if done."""
    plan = state.get("plan", [])
    index = state.get("step_index", 0)
    return plan[index] if 0 <= index < len(plan) else None


def draft_sql_node(state: TurnState, deps: AgentDeps) -> dict:
    """Ask the model for SQL for the current step, then validate it with the guard.

    Records the attempt either way. On success the step's `sql` is set and
    `execute_node` will run it next; on failure `sql` stays None and the repair
    budget is spent, which is what routes the graph back here for another try.
    """
    step = current_step(state)
    if step is None:
        return {}

    prompt = _prompt_for(state, deps, step)
    reply = deps.llm.invoke([HumanMessage(content=prompt)])
    sql = _strip_fences(message_text(reply))

    verdict = check_sql(
        sql,
        allowed_tables=deps.settings.allowed_tables,
        restricted_columns=deps.policy.restricted_columns(),
        default_limit=deps.settings.default_row_limit,
        max_limit=deps.settings.max_row_limit,
        qualify_with=deps.settings.bq_dataset,
    )

    attempts = list(state.get("sql_attempts", []))
    attempts.append(SqlAttempt(step_id=step.id, sql=sql, violations=verdict.violations))

    plan = list(state.get("plan", []))
    plan[state.get("step_index", 0)] = AnalysisStep(
        id=step.id,
        question=step.question,
        sql=verdict.sql if verdict.ok else None,
    )

    update: dict = {"plan": plan, "sql_attempts": attempts}
    if not verdict.ok:
        update["repair_budget"] = state.get("repair_budget", 0) - 1
    return update


def _prompt_for(state: TurnState, deps: AgentDeps, step: AnalysisStep) -> str:
    """Pick the repair prompt if the last attempt for this step failed, else the
    normal drafting prompt."""
    schema = render_schema(deps)
    attempts = state.get("sql_attempts", [])
    last = attempts[-1] if attempts else None

    if last is not None and last.step_id == step.id and last.failed:
        problem = "; ".join(last.violations) if last.violations else (last.error or "")
        return REPAIR_PROMPT.format(
            # A warehouse error describes the query the guard rewrote. Showing
            # the pre-rewrite draft alongside it asks for a fix to a query that
            # never ran; a guard violation has no rewrite, so it falls back.
            sql=last.executed_sql or last.sql,
            error=problem,
            question=step.question,
            schema=schema,
            dataset=deps.settings.bq_dataset,
        )

    return SQL_PROMPT.format(
        question=step.question,
        schema=schema,
        dataset=deps.settings.bq_dataset,
        prior_results=_prior_results(state),
        definitions=definitions_block(recalled(state, deps)),
        assumptions=sql_assumption_note(state.get("assumed_terms", [])),
    )


def _prior_results(state: TurnState) -> str:
    """Render earlier steps' results into text, so a later step in the same
    plan can build on them (e.g. joining against a number already fetched)."""
    frames = state.get("frames", {})
    if not frames:
        return ""
    rendered = "\n\n".join(
        f"Result of {key}:\n{frame.to_markdown(max_rows=5)}"
        for key, frame in frames.items()
    )
    return f"Results already gathered in this analysis:\n{rendered}"


def _strip_fences(text: str) -> str:
    """Remove a ```sql ... ``` wrapper if the model added one despite being told not to."""
    return FENCE.sub("", text).strip()
