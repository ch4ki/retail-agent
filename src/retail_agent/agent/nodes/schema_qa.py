"""Answers structural questions from cached schema metadata. No SQL runs."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.prompts import PERSONA_DEFAULT, SAFETY_RULES, SCHEMA_PROMPT
from retail_agent.agent.state import TurnState
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text


def render_schema(deps: AgentDeps) -> str:
    return "\n\n".join(schema.to_ddl() for schema in deps.source.describe_all())


def schema_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)
    prompt = SCHEMA_PROMPT.format(
        persona=PERSONA_DEFAULT,
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
