"""Intent classification. Falls back to `analyze` rather than failing a turn."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.prompts import ROUTER_PROMPT
from retail_agent.agent.state import TurnState
from retail_agent.llm.messages import message_text


log = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Which handler owns this turn.

    A schema rather than a word to match against a set: the provider constrains
    the model to these four values, so an unrecognised label cannot reach the
    graph and be silently rewritten to "analyze".
    """

    intent: Literal["schema", "analyze", "chat", "report_op"] = Field(
        description="schema: what data exists. analyze: needs a query. "
        "report_op: save, list or delete a saved report. "
        "chat: greetings, thanks, or a follow-up answerable from this "
        "conversation without new data."
    )

# A single answer can carry a whole result table. Enough to resolve "that" and
# "the same period", not enough for one verbose turn to dominate the prompt.
MAX_HISTORY_CHARS = 400

NO_HISTORY = "(this is the first question in the conversation)"


def last_user_message(state: TurnState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message_text(message).strip()
    return ""


def render_history(state: TurnState, limit: int) -> str:
    """Recent exchanges, oldest first, excluding the message being answered.

    Bounded twice — by message count and by characters per message — because
    history grows for the life of a session and every turn pays for it.
    """
    prior = state.get("messages", [])[:-1]
    if not prior:
        return NO_HISTORY

    lines = []
    for message in prior[-limit:]:
        who = "Executive" if isinstance(message, HumanMessage) else "Analyst"
        text = " ".join(message_text(message).split())
        if len(text) > MAX_HISTORY_CHARS:
            text = f"{text[:MAX_HISTORY_CHARS]}…"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def route_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)

    # A turn can arrive with no user message — Studio submits state directly.
    # Classifying nothing wastes a call and ends with the planner inventing a
    # step from an empty string.
    if not question:
        return {"intent": "chat"}

    prompt = ROUTER_PROMPT.format(
        history=render_history(state, deps.settings.history_messages)
    )
    try:
        decision = deps.llm.with_structured_output(RouteDecision).invoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)]
        )
    except Exception as err:
        # Not every model behind OpenRouter supports constrained decoding, and
        # a provider can still return something the schema rejects. Analysis is
        # the capable path, so guessing it costs a query rather than the turn.
        log.warning("router structured output failed (%s); assuming analyze", err)
        return {"intent": "analyze"}

    return {"intent": decision.intent}
