"""Decomposes a question into one or more single-query steps."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.nodes.schema_qa import render_schema
from retail_agent.agent.prompts import PLANNER_PROMPT
from retail_agent.agent.state import AnalysisStep, TurnState
from retail_agent.llm.messages import message_text


def plan_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)
    max_steps = deps.settings.max_analysis_steps

    prompt = PLANNER_PROMPT.format(max_steps=max_steps, schema=render_schema(deps))
    reply = deps.llm.invoke([HumanMessage(content=f"{prompt}\n\nQuestion: {question}")])

    questions = _parse_steps(message_text(reply))[:max_steps]
    if not questions:
        questions = [question]

    plan = [
        AnalysisStep(id=f"step_{i + 1}", question=text)
        for i, text in enumerate(questions)
    ]
    return {"plan": plan, "step_index": 0}


def _parse_steps(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("STEP:"):
            content = stripped[5:].strip()
            if content:
                steps.append(content)
    return steps
