"""Rendering the warehouse's shape for a prompt.

Two renderings, because callers want different things and only one of them
should pay for the expensive one. `render_schema_outline` is the tables and
their columns; `render_schema_for_sql` adds the values each enumerable column
actually holds, which costs a metadata scan per table and is only worth it to
something about to write a literal into a WHERE clause.
"""

from __future__ import annotations

import logging
import time

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps, TurnContext
from retail_agent.agent.prompts import SCHEMA_PROMPT
from retail_agent.agent.state import step_event
from retail_agent.datasources.column_values import enumerable_columns, with_values
from retail_agent.knowledge.conventions import notes_for

log = logging.getLogger(__name__)


def build_schema_tool(deps: AgentDeps, capture: TurnCapture) -> list[BaseTool]:
    """"What data do you have?" answered without spending anything.

    A tool rather than a subagent: there is no loop to run and nothing to
    decide. It hands back the structure and lets the supervisor put it in its
    own words, which is also what keeps the persona applied to the answer.
    """

    @tool
    def describe_schema(runtime: ToolRuntime[TurnContext, object]) -> Command:
        """Describe the data available: tables, columns and what they support.

        Runs no query and costs nothing. Use this for questions about what can
        be asked rather than about the numbers themselves.
        """
        started = time.perf_counter()
        with capture.step("describe_schema") as step:
            schemas = deps.source.describe_all()
            step.detail = f"{len(schemas)} table(s)"
            answer = SCHEMA_PROMPT.format(
                schema="\n\n".join(schema.to_ddl() for schema in schemas)
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=answer, tool_call_id=runtime.tool_call_id)
                    ],
                    "events": [step_event("describe_schema", started, step.detail)],
                    "calls": 1,
                }
            )

    return [describe_schema]


def render_schema_outline(deps: AgentDeps) -> str:
    """Tables and columns, and nothing that costs a round trip.

    Enough to keep a prompt from naming a column that does not exist, which is
    all a caller writing prose rather than SQL can use.
    """
    return "\n\n".join(schema.to_ddl() for schema in deps.source.describe_all())


def render_schema_for_sql(deps: AgentDeps) -> str:
    """The same schema, plus the values each enumerable column actually holds.

    Only the analyst needs this, and only the analyst should pay for it. A bare
    `gender STRING` cannot stop the model writing `gender = 'female'` against a
    column holding 'F' — that happened twice in one eval run, and both queries
    were valid, passed the guard, ran without error and returned zero rows, so
    no layer reported a problem.
    """
    schemas = deps.source.describe_all()
    values = _discover_values(deps, schemas)
    restricted = {name.lower() for name in deps.policy.restricted_columns()}
    return "\n\n".join(
        with_values(
            schema,
            values.get(schema.name, {}),
            # Same rule as the values: nothing is said about a PII column, so a
            # note added carelessly cannot end up describing one.
            notes={
                column: note
                for column, note in notes_for(schema.name).items()
                if column.lower() not in restricted
            },
        ).to_ddl()
        for schema in schemas
    )


def _discover_values(deps: AgentDeps, schemas) -> dict[str, dict[str, tuple[str, ...]]]:
    """Each table's enumerable values, from the source.

    Asks for a dedicated `column_values` method rather than going through
    `execute`: that path is for the user's guarded, dry-run, cost-capped
    queries, and an internal metadata scan should not consume that budget or
    appear in the turn's SQL attempts. A source without the method — every test
    double — simply yields no values.

    Caching belongs to the source, which already caches schemas.
    """
    lookup = getattr(deps.source, "column_values", None)
    if lookup is None:
        return {}

    restricted = deps.policy.restricted_columns()
    discovered: dict[str, dict[str, tuple[str, ...]]] = {}

    for schema in schemas:
        columns = enumerable_columns(schema, restricted=restricted)
        if not columns:
            continue
        try:
            discovered[schema.name] = lookup(schema.name, columns)
        except Exception as err:
            # Best effort. A warehouse that refuses this should cost the hint,
            # not the turn.
            log.warning("could not read column values for %s (%s)", schema.name, err)

    return discovered
