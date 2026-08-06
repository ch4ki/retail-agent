"""Decomposes a question into one or more single-query steps."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message, render_history
from retail_agent.agent.nodes.schema_qa import render_schema
from retail_agent.agent.prompts import PLANNER_PROMPT
from retail_agent.agent.state import AnalysisStep, TurnState


log = logging.getLogger(__name__)


class Plan(BaseModel):
    """The retrieval steps a question decomposes into.

    One artifact instead of two. The prompt and a hand-written parser used to be
    separate things that could disagree, and when they did — a planner told to
    emit SQL while the parser looked for `STEP:` lines — every question silently
    collapsed to a single step.
    """

    steps: list[str] = Field(
        description="Each step is one retrieval, phrased so it stands alone and "
        "could be turned into a single query. Never a comparison or an "
        "explanation of earlier steps."
    )


def plan_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)
    if not question:
        return {}

    max_steps = deps.settings.max_analysis_steps

    prompt = PLANNER_PROMPT.format(
        max_steps=max_steps,
        schema=render_schema(deps),
        history=render_history(state, deps.settings.history_messages),
    )
    try:
        decomposed = deps.llm.with_structured_output(Plan).invoke(
            [HumanMessage(content=f"{prompt}\n\nQuestion: {question}")]
        )
        steps = decomposed.steps
    except Exception as err:
        # Constrained decoding is not universal across OpenRouter models. One
        # step carrying the raw question still answers most questions.
        log.warning("planner structured output failed (%s); using one step", err)
        steps = []

    # An empty list is possible even on success; asking the original question as
    # a single step beats answering nothing.
    questions = [text for text in steps if text.strip()][:max_steps]
    if not questions:
        log.warning("planner returned no steps for %r; using it verbatim", question)
        questions = [question]

    plan = [
        AnalysisStep(id=f"step_{i + 1}", question=text)
        for i, text in enumerate(questions)
    ]
    return {"plan": plan}
