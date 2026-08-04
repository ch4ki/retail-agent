"""Intent classification. Falls back to `analyze` rather than failing a turn."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import ROUTER_PROMPT
from retail_agent.agent.state import TurnState
from retail_agent.llm.messages import message_text

VALID_INTENTS = {"schema", "analyze", "chat"}


def last_user_message(state: TurnState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def route_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)
    reply = deps.llm.invoke(
        [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=question)]
    )
    label = message_text(reply).lower().strip(".")

    return {"intent": label if label in VALID_INTENTS else "analyze"}
