"""Runs guarded SQL and masks the result before it can enter model context."""

from __future__ import annotations

import logging

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.sql import current_step
from retail_agent.agent.state import MaskedFrame, SqlAttempt, TurnState
from retail_agent.datasources.base import DataSourceError
from retail_agent.safety.pii import mask_dataframe

log = logging.getLogger(__name__)


def execute_node(state: TurnState, deps: AgentDeps) -> dict:
    step = current_step(state)
    if step is None or not step.sql:
        return {"repair_budget": state.get("repair_budget", 0) - 1}

    attempts = list(state.get("sql_attempts", []))

    try:
        deps.source.assert_within_budget(step.sql)
        result = deps.source.execute(step.sql)
    except DataSourceError as err:
        log.info("query failed for %s: %s", step.id, err)
        attempts.append(SqlAttempt(step_id=step.id, sql=step.sql, error=str(err)))
        return {
            "sql_attempts": attempts,
            "repair_budget": state.get("repair_budget", 0) - 1,
        }

    masked, report = mask_dataframe(result.rows, deps.policy, salt=deps.settings.pii_salt)

    frames = dict(state.get("frames", {}))
    frames[step.id] = MaskedFrame(
        key=step.id,
        frame=masked,
        row_count=result.row_count,
        redactions=report.redactions,
        dropped_columns=report.dropped_columns,
    )

    attempts.append(
        SqlAttempt(
            step_id=step.id,
            sql=step.sql,
            row_count=result.row_count,
            bytes_billed=result.bytes_billed,
        )
    )

    return {
        "frames": frames,
        "sql_attempts": attempts,
        "step_index": state.get("step_index", 0) + 1,
        "redactions": state.get("redactions", 0) + report.redactions,
    }
