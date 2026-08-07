"""Answers structural questions from cached schema metadata. No SQL runs."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.prompts import SAFETY_RULES, SCHEMA_PROMPT
from retail_agent.agent.state import TurnState
from retail_agent.knowledge.column_values import (
    build_discovery_query,
    enumerable_columns,
    read_discovery_row,
    with_values,
)
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text
from retail_agent.store.personas import active_body

log = logging.getLogger(__name__)


def render_schema(deps: AgentDeps) -> str:
    """Structure only: table and column names with their types.

    Deliberately free of any warehouse query. `schema_node` answers "what data
    do you have" from this, and that path is asserted to cost nothing — see
    `test_schema_node_answers_without_sql`.
    """
    return "\n\n".join(schema.to_ddl() for schema in deps.source.describe_all())


def render_schema_for_sql(deps: AgentDeps) -> str:
    """The same schema, plus the values each enumerable column actually holds.

    Only the SQL-drafting path needs this, and only that path should pay for it.
    A bare `gender STRING` cannot stop the model writing `gender = 'female'`
    against a column holding 'F' — that happened twice in one eval run, and both
    queries were valid, passed the guard, ran without error and returned zero
    rows, so no layer reported a problem.
    """
    schemas = deps.source.describe_all()
    values = _discover_values(deps, schemas)
    return "\n\n".join(
        with_values(schema, values.get(schema.name, {})).to_ddl() for schema in schemas
    )


def _discover_values(deps: AgentDeps, schemas) -> dict[str, dict[str, tuple[str, ...]]]:
    """Read each table's enumerable values once per process.

    Cached on the source: the SQL prompt is built on every analysis turn, and
    paying for a warehouse scan each time would be a recurring cost for a fact
    that does not change within a session.

    Asks the source for a dedicated `column_values` method rather than going
    through `execute`. That path is for the user's guarded, dry-run, budgeted
    queries; borrowing it for internal metadata would make every double serve
    both, and would put an internal scan inside the user's cost ceiling.
    """
    cached = getattr(deps.source, "_column_values_cache", None)
    if cached is not None:
        return cached

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

    try:
        deps.source._column_values_cache = discovered
    except AttributeError:
        pass  # a frozen or slotted double; recompute next time rather than fail
    return discovered


def schema_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)
    prompt = SCHEMA_PROMPT.format(
        persona=active_body(deps.personas),
        safety=SAFETY_RULES,
        question=question,
        schema=render_schema(deps),
    )
    reply = deps.llm.invoke([HumanMessage(content=prompt)])
    scanned = scan_text(message_text(reply))

    return {
        "answer": scanned.text,
        "status": "ok",
        "messages": [AIMessage(content=scanned.text)],
    }
