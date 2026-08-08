"""Drafts SQL and puts it through the guard before anything can execute it."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.recall import definitions_for
from retail_agent.agent.nodes.schema_qa import render_schema_for_sql
from retail_agent.agent.prompts import REPAIR_PROMPT, SQL_PROMPT
from retail_agent.agent.state import AnalysisStep, MaskedFrame, SqlAttempt, TurnState
from retail_agent.knowledge.trios import sql_assumption_note
from retail_agent.llm.messages import message_text
from retail_agent.safety.sql_guard import check_sql, without_limit

FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

# Kept out of the function so the composition logic reads as logic rather than
# as a wall of prose.
COMPOSITION_RULE = (
    "Write ONE self-contained query. Where you need the rows above, repeat or "
    "nest the query that produced them — never copy result values into a "
    "literal, and never write a placeholder comment for rows you cannot see. "
    "Only the single-row values quoted above may be used directly."
)


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
        # Both are the safety ceiling. The SQL-level bound exists only to stop
        # an unbounded result; how many rows are shown is decided when the
        # result is read, so that `total_rows` reports the true size.
        default_limit=deps.settings.max_row_limit,
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
    schema = render_schema_for_sql(deps)
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
        definitions=_all_definitions(state, deps),
        assumptions=sql_assumption_note(state.get("assumed_terms", [])),
    )


def _prior_results(state: TurnState) -> str:
    """What a later step is told about the steps before it.

    This used to render each earlier frame as markdown truncated to five rows,
    headed "Results already gathered". That produced fabricated answers: the
    model has no way to *reference* those rows from SQL, so it copied the ones
    it could see into a literal and left a comment asking for the rest —

        SELECT AVG(age) FROM (SELECT 54 AS age UNION ALL SELECT 25 ...)
        /* Add the remaining 95 rows here */

    — which passed the guard, ran cleanly and answered confidently from five
    sampled rows out of a hundred.

    So the rule is structural rather than an instruction: a multi-row result is
    described by its *query*, never its rows. A model cannot inline data it was
    never shown. A single row is different — it cannot be truncated, and
    "the average is 41, now find who is above it" genuinely needs the number —
    so those are still passed as values.
    """
    frames = state.get("frames", {})
    if not frames:
        return ""

    queries = {
        step.id: step.sql for step in state.get("plan", []) if getattr(step, "sql", None)
    }
    blocks = [
        _describe(step_id, frame, queries.get(step_id))
        for step_id, frame in frames.items()
    ]

    return (
        "Results already gathered in this analysis:\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + COMPOSITION_RULE
    )


def _describe(step_id: str, frame: MaskedFrame, query: str | None) -> str:
    """One earlier step, as the next one is allowed to see it."""
    if len(frame.rows) == 1 and frame.row_count == 1:
        values = ", ".join(
            f"{column} = {value!r}" for column, value in zip(frame.columns, frame.rows[0])
        )
        return f"{step_id} returned exactly one row: {values}"

    # `row_count` is the true size of the result even when only some rows were
    # fetched, so this is exact rather than the hedge it used to be.
    how_many = (
        f"{frame.row_count} rows (only the first {len(frame.rows)} shown)"
        if frame.truncated
        else f"{frame.row_count} rows"
    )
    columns = ", ".join(frame.columns)
    # No recorded query means there is nothing to compose against, and naming
    # the step anyway would invite an invented table reference.
    provenance = f" Its query was:\n{without_limit(query)}" if query else ""
    return f"{step_id} returned {how_many} of ({columns}).{provenance}"


def _strip_fences(text: str) -> str:
    """Remove a ```sql ... ``` wrapper if the model added one despite being told not to."""
    return FENCE.sub("", text).strip()


# One definition of "the definitions", in `recall`, which is the node that
# resolves them. It lived here until the planner turned out to need it too, and
# a second copy is how the two would drift into disagreeing about a term.
_all_definitions = definitions_for
