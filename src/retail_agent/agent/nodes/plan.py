"""Decomposes a question into one or more single-query steps."""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.recall import definitions_for
from retail_agent.agent.nodes.route import last_user_message, render_history
from retail_agent.agent.nodes.schema_qa import render_schema
from retail_agent.agent.prompts import PLANNER_PROMPT
from retail_agent.agent.state import AnalysisStep, TurnState


log = logging.getLogger(__name__)


# A step whose purpose is to look something up rather than query it. `recall`
# has already done that: whatever the analysts defined is in the prompt, and
# whatever they did not is stated as an assumption. Observed live twice — the
# second time after the prompt was explicitly told not to, which is why this is
# a filter in code rather than another sentence asking nicely.
_LOOKUP_STEP = re.compile(
    r"\b(?:retrieve|find|look\s*up|determine|obtain|identify|clarify)\b"
    r"[^.]*\b(?:definition|criteria|documentation|business\s+context)\b",
    re.IGNORECASE,
)


def is_lookup_step(text: str) -> bool:
    """True for a step that cannot become a query.

    Narrow on purpose: "count customers matching the churn definition" is real
    work and must survive. Only a step whose verb is a lookup *and* whose
    object is a definition is dropped.
    """
    return bool(_LOOKUP_STEP.search(text))


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
        # `recall` resolved these before this node ran, and until they were
        # passed here the planner invented its own reading of every business
        # term — leaving `draft_sql` a step from which the defined word had
        # already been removed.
        definitions=definitions_for(state, deps),
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
    usable = []
    seen = set()
    for text in steps:
        if not text.strip():
            continue
        if is_lookup_step(text):
            log.info("dropping non-retrieval step: %r", text)
            continue
        # Observed live: the same sentence returned twice for one question.
        # That runs the query twice and lets `_final_frame` score the repeat as
        # though it were a separate finding.
        key = " ".join(text.split()).lower()
        if key in seen:
            log.info("dropping duplicate step: %r", text)
            continue
        seen.add(key)
        usable.append(text)

    questions = usable[:max_steps]
    if not questions:
        log.warning("planner returned no steps for %r; using it verbatim", question)
        questions = [question]

    plan = [
        AnalysisStep(id=f"step_{i + 1}", question=text)
        for i, text in enumerate(questions)
    ]
    return {"plan": plan}
