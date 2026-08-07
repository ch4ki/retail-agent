"""Runs guarded SQL and masks the result before it can enter model context."""

from __future__ import annotations

import logging
from dataclasses import replace

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.sql import current_step
from retail_agent.agent.state import AnalysisStep, MaskedFrame, SqlAttempt, TurnState
from retail_agent.datasources.base import DataSourceError
from retail_agent.safety.pii import mask_dataframe
from retail_agent.safety.sql_guard import applied_limit

log = logging.getLogger(__name__)


def execute_node(state: TurnState, deps: AgentDeps) -> dict:
    step = current_step(state)
    if step is None or not step.sql:
        return {"repair_budget": state.get("repair_budget", 0) - 1}

    attempts = state.get("sql_attempts", [])

    try:
        deps.source.assert_within_budget(step.sql)
        result = deps.source.execute(step.sql)
    except DataSourceError as err:
        log.info("query failed for %s: %s", step.id, err)
        return {
            "sql_attempts": _completed(attempts, step, error=str(err)),
            "repair_budget": state.get("repair_budget", 0) - 1,
        }

    masked, report = mask_dataframe(result.rows, deps.policy, salt=deps.settings.pii_salt)

    frames = dict(state.get("frames", {}))
    # A result that exactly fills its cap almost certainly had more behind it.
    # There is no way to tell from the rows themselves, and the cost of the
    # false positive — saying "at least 500" when there were exactly 500 — is
    # nothing next to reporting a truncated sample as a total.
    cap = applied_limit(step.sql)
    frames[step.id] = MaskedFrame.from_dataframe(
        masked,
        row_count=result.row_count,
        redactions=report.redactions,
        dropped_columns=report.dropped_columns,
        truncated=cap is not None and result.row_count >= cap,
    )

    return {
        "frames": frames,
        "sql_attempts": _completed(
            attempts,
            step,
            row_count=result.row_count,
            bytes_billed=result.bytes_billed,
        ),
        "step_index": state.get("step_index", 0) + 1,
        "redactions": state.get("redactions", 0) + report.redactions,
    }


def _completed(
    attempts: list[SqlAttempt], step: AnalysisStep, **outcome
) -> list[SqlAttempt]:
    """Fold this run's outcome into the record `draft_sql` already opened.

    Appending instead would count one query as two tries. The fallback covers a
    node invoked without a preceding draft, which only happens in tests.
    """
    attempts = list(attempts)
    draft = attempts[-1] if attempts else None

    if draft is not None and draft.step_id == step.id and not draft.failed:
        attempts[-1] = replace(draft, executed_sql=step.sql, **outcome)
    else:
        attempts.append(
            SqlAttempt(
                step_id=step.id, sql=step.sql, executed_sql=step.sql, **outcome
            )
        )
    return attempts
