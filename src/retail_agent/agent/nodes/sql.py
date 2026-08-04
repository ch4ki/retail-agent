"""Drafts SQL and puts it through the guard before anything can execute it."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.schema_qa import render_schema
from retail_agent.agent.prompts import REPAIR_PROMPT, SQL_PROMPT
from retail_agent.agent.state import AnalysisStep, SqlAttempt, TurnState
from retail_agent.llm.messages import message_text
from retail_agent.safety.sql_guard import check_sql

FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def current_step(state: TurnState) -> AnalysisStep | None:
    plan = state.get("plan", [])
    index = state.get("step_index", 0)
    return plan[index] if 0 <= index < len(plan) else None


def draft_sql_node(state: TurnState, deps: AgentDeps) -> dict:
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
        result_key=step.result_key,
    )

    update: dict = {"plan": plan, "sql_attempts": attempts}
    if not verdict.ok:
        update["repair_budget"] = state.get("repair_budget", 0) - 1
    return update


def _prompt_for(state: TurnState, deps: AgentDeps, step: AnalysisStep) -> str:
    schema = render_schema(deps)
    attempts = state.get("sql_attempts", [])
    last = attempts[-1] if attempts else None

    if last is not None and last.step_id == step.id and last.failed:
        problem = "; ".join(last.violations) if last.violations else (last.error or "")
        return REPAIR_PROMPT.format(
            sql=last.sql, error=problem, question=step.question, schema=schema
        )

    return SQL_PROMPT.format(
        question=step.question,
        schema=schema,
        dataset=deps.settings.bq_dataset,
        prior_results=_prior_results(state),
    )


def _prior_results(state: TurnState) -> str:
    frames = state.get("frames", {})
    if not frames:
        return ""
    rendered = "\n\n".join(
        f"Result of {key}:\n{frame.to_markdown(max_rows=5)}"
        for key, frame in frames.items()
    )
    return f"Results already gathered in this analysis:\n{rendered}"


def _strip_fences(text: str) -> str:
    return FENCE.sub("", text).strip()
