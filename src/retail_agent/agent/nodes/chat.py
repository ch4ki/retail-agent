"""Conversational follow-ups answered from history, without touching the warehouse."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import CHAT_PROMPT, PERSONA_DEFAULT, SAFETY_RULES
from retail_agent.agent.state import TurnState
from retail_agent.safety.egress import scan_text


def chat_node(state: TurnState, deps: AgentDeps) -> dict:
    system = SystemMessage(
        content=CHAT_PROMPT.format(persona=PERSONA_DEFAULT, safety=SAFETY_RULES)
    )
    reply = deps.llm.invoke([system, *state.get("messages", [])])
    scanned = scan_text(str(reply.content))

    return {
        "answer": scanned.text,
        "status": "ok",
        "messages": [AIMessage(content=scanned.text)],
    }
