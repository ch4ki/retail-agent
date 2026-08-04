"""Conversational follow-ups answered from history, without touching the warehouse."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.prompts import CHAT_PROMPT, PERSONA_DEFAULT, SAFETY_RULES
from retail_agent.agent.state import TurnState
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text


BLANK_TURN_REPLY = (
    "I didn't catch a question. What would you like to know about the sales, "
    "product or customer data?"
)


def chat_node(state: TurnState, deps: AgentDeps) -> dict:
    if not last_user_message(state):
        return {
            "answer": BLANK_TURN_REPLY,
            "status": "ok",
            "messages": [AIMessage(content=BLANK_TURN_REPLY)],
        }

    system = SystemMessage(
        content=CHAT_PROMPT.format(persona=PERSONA_DEFAULT, safety=SAFETY_RULES)
    )
    reply = deps.llm.invoke([system, *state.get("messages", [])])
    scanned = scan_text(message_text(reply))

    return {
        "answer": scanned.text,
        "status": "ok",
        "messages": [AIMessage(content=scanned.text)],
    }
