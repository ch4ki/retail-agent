"""The two tools the ReAct arm is given.

Both are wrappers. `run_sql` calls the same `check_sql` and `mask_dataframe`
that `draft_sql_node` and `execute_node` call, with the same settings, so a
query the graph would reject is rejected here by the same code rather than by a
second implementation that could drift from it. That is what makes the
head-to-head a comparison of control flow instead of a comparison of guards.

What is deliberately *not* here is any ordering between them. `lookup_definitions`
is available, not compulsory — the graph runs `recall_node` on every analyze turn
whether the model wants it or not, and whether that guarantee is worth its
rigidity is the question this baseline exists to answer.
"""

from __future__ import annotations

from collections.abc import Callable

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.state import MaskedFrame
from retail_agent.baseline.capture import ResultCapture
from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.trios import definitions_block, live_trios
from retail_agent.safety.pii import mask_dataframe
from retail_agent.safety.sql_guard import check_sql

NOTHING_DEFINED = (
    "No agreed definition covers this question. Decide for yourself, and say "
    "in your answer which interpretation you used."
)


class GuardRejection(Exception):
    """The query never ran. Carries the violations so `ToolErrorMiddleware` can
    hand them back to the model as the repair prompt does on the graph arm."""


def build_tools(deps: AgentDeps, capture: ResultCapture) -> list[Callable]:
    """Tools bound to one turn's capture.

    Closures rather than methods so that `create_agent` receives plain callables
    with docstrings — which is the tool description the model actually reads.
    """

    def run_sql(sql: str) -> str:
        """Run a read-only BigQuery query against theLook and return the rows.

        Write exactly one SELECT statement. Do not add a LIMIT; one is applied
        for you. If the query is rejected or fails you will be told why and may
        try again.
        """
        verdict = check_sql(
            sql,
            allowed_tables=deps.settings.allowed_tables,
            restricted_columns=deps.policy.restricted_columns(),
            default_limit=deps.settings.max_row_limit,
            max_limit=deps.settings.max_row_limit,
            qualify_with=deps.settings.bq_dataset,
        )
        if not verdict.ok:
            raise GuardRejection("; ".join(verdict.violations))

        deps.source.assert_within_budget(verdict.sql)
        result = deps.source.execute(verdict.sql)

        masked, report = mask_dataframe(
            result.rows, deps.policy, salt=deps.settings.pii_salt
        )
        frame = MaskedFrame.from_dataframe(
            masked,
            row_count=result.row_count,
            redactions=report.redactions,
            dropped_columns=report.dropped_columns,
            truncated=result.row_count > len(masked),
        )
        capture.record_query(verdict.sql, frame)
        return _render(frame)

    def lookup_definitions(question: str) -> str:
        """Look up how the business defines the terms in a question.

        Use this before writing SQL whenever a question contains a term whose
        meaning is a business decision rather than a column — loyal, churn, top,
        engaged, at risk, high value, underspending, performing well.
        """
        found = retrieve(
            question,
            live_trios(deps.trios),
            dense_rank=deps.dense.rank if deps.dense is not None else None,
        )
        capture.record_definitions([trio.id for trio in found])

        block = definitions_block(found)
        # An empty string reads to a model as a definition of nothing, and it
        # will proceed as though the term were settled. Saying so is the whole
        # protection the graph gets from its undefined-term branch.
        return block.strip() or NOTHING_DEFINED

    return [run_sql, lookup_definitions]


def _render(frame: MaskedFrame) -> str:
    """What the model sees of a result.

    `to_markdown` already caps the rows it prints and says how many were
    withheld, so a truncated result cannot read as a complete one.
    """
    body = frame.to_markdown()
    if frame.dropped_columns:
        body += (
            f"\n\n_(withheld by policy: {', '.join(frame.dropped_columns)})_"
        )
    return body
