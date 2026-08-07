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

    # Folded into the routing call rather than given a call of its own: the
    # model is already reading this question, so detection costs no extra
    # request and no extra latency.
    style_field: Literal["depth", "answer_format", "none"] = Field(
        default="none",
        description="Whether the user said anything about HOW answers should be "
        "presented, as opposed to what they are asking about. depth: how much "
        "reasoning they want. answer_format: how it should be laid out. "
        "A question about the data ('why are sales down?') is not a preference "
        "— use 'none'.",
    )
    style_value: str = Field(
        default="",
        description="For depth: 'summary' if they want less, 'deep' if they "
        "want more reasoning. For answer_format: 'table', 'bullets' or 'prose'. "
        "Wanting a shorter or longer ANSWER is depth, not answer_format.",
    )
    style_evidence: str = Field(
        default="",
        description="The span of the question showing this, copied VERBATIM. "
        "Leave empty if there is no such span.",
    )


# Which values mean something for each field. `style_value` has to be free text
# in the schema because the legal set depends on `style_field`, so it is checked
# here instead of being trusted.
STYLE_VALUES: dict[str, frozenset[str]] = {
    "depth": frozenset({"summary", "deep"}),
    "answer_format": frozenset({"table", "bullets", "prose"}),
}


def style_signal(decision: RouteDecision, *, question: str) -> "Signal | None":
    """The preference the model saw in this question, if it can be evidenced.

    The model is trusted to understand the sentence — including negation, which
    is where the regex this replaced went wrong, recording `depth=summary` for
    "don't just give me the number, tell me why". It is not trusted with the
    quote: the evidence must appear in the question, or the signal is dropped.

    That is what keeps the eventual proposal honest. It says "you asked for this
    three times, most recently '<span>'", and a span the user never typed would
    make that a fabrication rather than a citation.
    """
    from retail_agent.store.learning import Signal

    if decision.style_field == "none":
        return None
    if decision.style_value not in STYLE_VALUES.get(decision.style_field, frozenset()):
        return None

    evidence = decision.style_evidence.strip()
    # No quote means nothing to show the user, and an unquotable suggestion is
    # the kind this design refuses to make.
    if not evidence or evidence.lower() not in question.lower():
        return None

    return Signal(field=decision.style_field, value=decision.style_value, evidence=evidence)


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

    _remember_style(decision, question=question, state=state, deps=deps)
    return {"intent": decision.intent}


def _remember_style(decision, *, question: str, state: TurnState, deps: AgentDeps) -> None:
    """Accumulate evidence about how this user likes answers presented.

    Here rather than in the CLI, which is where it used to live: the detection
    is part of this node's model call now, and leaving the write outside would
    mean the router finds a signal and hands it to someone else to store — so
    Studio, the eval harness and any other caller would learn nothing.

    Nothing is applied from this. Evidence accumulates and the agent eventually
    *asks*; see `store.learning.next_proposal`.
    """
    if deps.signals is None:
        return

    signal = style_signal(decision, question=question)
    if signal is None:
        return

    try:
        deps.signals.record(user_id=state.get("user_id", ""), signal=signal)
    except Exception as err:
        # A learning write is worth less than an answer, always.
        log.debug("signal not recorded: %s", err)
