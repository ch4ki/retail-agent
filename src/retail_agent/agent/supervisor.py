"""The agent, assembled.

One `create_agent` with ten tools and one middleware stack. This is the whole
control flow of the product; there is no second place a turn can be shaped.

The system prompt is absent on purpose — `supervisor_middleware` supplies it
per model call, so an edited persona and a changed preference take effect on
the next turn rather than the next deploy.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from retail_agent.agent.deps import AgentDeps, TurnContext
from retail_agent.agent.memory import build_memory_tools
from retail_agent.agent.middleware import supervisor_middleware
from retail_agent.agent.reports import build_report_tools
from retail_agent.agent.schema import build_schema_tool
from retail_agent.agent.state import TurnState
from retail_agent.agent.subagents import build_subagents


def build_tools(
    deps: AgentDeps, *, pause_for_definitions: bool = False
) -> list[BaseTool]:
    """Every capability the agent has, in one list.

    Adding one is a `@tool` in the relevant builder — that is the extensibility
    story, and it is the same either way, whether the new capability is a plain
    function or another `create_agent` behind one.
    """
    return [
        *build_subagents(deps),
        *build_schema_tool(deps),
        *build_report_tools(deps),
        *build_memory_tools(deps, pause_for_definitions=pause_for_definitions),
    ]


def build_agent(
    deps: AgentDeps,
    checkpointer=None,
    *,
    pause_for_definitions: bool = False,
):
    """The compiled agent for one turn.

    What the tools do is written into checkpointed graph state (`TurnState`)
    rather than into anything the caller builds and hands in, and that state
    is the unit the eval scores and the trace records.

    `pause_for_definitions` says there is a person on the other end who can
    settle a term the model asked about. Only the CLI passes it. A headless
    caller cannot answer a pause, so for those the `ask_for_definitions` tool
    runs through to its body, which records the assumption and requires the
    answer to disclose it.
    """
    return create_agent(
        model=deps.llm,
        tools=build_tools(deps, pause_for_definitions=pause_for_definitions),
        middleware=supervisor_middleware(deps),
        checkpointer=checkpointer,
        context_schema=TurnContext,
        state_schema=TurnState,
    )
