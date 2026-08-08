"""The ReAct arm: `create_agent` plus the middleware that makes it a fair test.

A bare ReAct loop would be a straw man. `langchain.agents.middleware` covers
more of the graph's behaviour than the framing "graph versus agent" suggests —
`ToolCallLimitMiddleware` scopes a call budget to one tool, `ToolErrorMiddleware`
hands a failure back to the model the way the repair edge does. Using them is
what makes a loss here mean something.

What has no stock counterpart, and is therefore left out:

- **`diagnose_node`.** `ToolRetryMiddleware` retries on exceptions, and a
  zero-row result raises nothing — it is a successful call. `WHERE brand =
  'Levis'` against a stored `Levi's` returns cleanly, and the agent reports no
  revenue. If those cases fail on this arm, the measurement is working.
- **`recall_node`'s guarantee.** `lookup_definitions` is offered, not forced.
- **`PLANNER_PROMPT`.** Deliberately absent from the system prompt: whether to
  decompose a question, and whether to revisit that after seeing rows, is the
  thing this loop is claimed to do better. Instructing it would pre-empt the
  answer.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.schema_qa import render_schema_for_sql
from retail_agent.baseline.capture import ResultCapture
from retail_agent.baseline.tools import GuardRejection, build_tools
from retail_agent.config import Settings
from retail_agent.datasources.base import DataSourceError

# Redacted rather than blocked: blocking raises and kills the turn, and a case
# that dies is scored as an agent failure rather than as the leak it was.
# `apply_to_tool_results` is the setting that matters — PII arrives from the
# warehouse inside a ToolMessage, never from the executive's question.
PII_TYPES = ("email", "credit_card", "ip")

# A model that has stopped making progress still costs money on every turn.
MAX_MODEL_CALLS = 30

SYSTEM_PROMPT = """\
You are a data analyst answering questions about theLook, a retail dataset in \
BigQuery.

Use `run_sql` to query it. Use `lookup_definitions` when a question contains a \
term whose meaning is a business decision rather than a column.

{schema}

Write standard BigQuery SQL. Fully qualify tables as `{dataset}.table`. Do not \
add a LIMIT; one is applied for you. Answer with the number you found and one \
sentence of context.\
"""


def build_middleware(settings: Settings) -> list[AgentMiddleware]:
    """The stack, derived from the same settings the graph reads.

    `run_limit` is computed rather than written down so that tuning a budget in
    config moves both arms together. A ReAct arm silently allowed twice the
    queries would win on accuracy for a reason the report would not mention.
    """
    sql_budget = (
        settings.max_analysis_steps + settings.repair_budget + _DIAGNOSE_BUDGET
    )

    return [
        *(
            PIIMiddleware(
                pii_type,
                strategy="redact",
                apply_to_input=False,
                apply_to_tool_results=True,
            )
            for pii_type in PII_TYPES
        ),
        ToolCallLimitMiddleware(tool_name="run_sql", run_limit=sql_budget),
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ToolErrorMiddleware(on_error=_describe_failure),
    ]


# `fresh_scratch` defaults this and the graph spends it outside `repair_budget`.
# Named here rather than imported because it is a default argument there, not a
# setting — if it becomes one, this should follow it.
_DIAGNOSE_BUDGET = 1


def _describe_failure(error: Exception, request: object) -> str | None:
    """Turn a tool failure into the message the model gets back.

    This is the ReAct arm's repair prompt. Only the two failures the graph also
    recovers from are handled; anything else propagates, so an internal bug
    surfaces as a failed case rather than as an agent that quietly worked around
    it.

    `request` is the `ToolCallRequest` and is unused, but the parameter is not
    optional: `OnError` is `Callable[[Exception, ToolCallRequest], ...]`, and a
    one-argument version type-checks, imports, and passes every test that does
    not call it — then raises TypeError against the live provider the first time
    a query is rejected. Which is exactly what happened: the repair path was
    dead for a whole eval run and reported as an agent failure.
    """
    if isinstance(error, GuardRejection):
        return f"The query was rejected before running: {error}. Rewrite it."
    if isinstance(error, DataSourceError):
        return f"The query failed: {error}. Fix it and try again."
    return None


def build_react_agent(deps: AgentDeps, capture: ResultCapture, checkpointer=None):
    """The compiled ReAct agent for one turn.

    Bound to a capture because the tools write their results into it; a turn is
    the unit both arms are scored on.
    """
    return create_agent(
        model=deps.llm,
        tools=build_tools(deps, capture),
        system_prompt=SYSTEM_PROMPT.format(
            schema=render_schema_for_sql(deps), dataset=deps.settings.bq_dataset
        ),
        middleware=build_middleware(deps.settings),
        checkpointer=checkpointer,
    )
