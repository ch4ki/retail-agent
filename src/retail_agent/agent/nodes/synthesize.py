"""Turns masked results into prose, then sweeps the prose for leaked PII."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.nodes.route import last_user_message
from retail_agent.agent.prompts import SAFETY_RULES, SYNTHESIS_PROMPT
from retail_agent.agent.state import TurnState
from retail_agent.llm.messages import message_text
from retail_agent.safety.egress import scan_text
from retail_agent.store.personas import active_body
from retail_agent.store.preferences import preferred, style_instruction


def synthesize_node(state: TurnState, deps: AgentDeps) -> dict:
    question = last_user_message(state)
    frames = state.get("frames", {})

    if not frames:
        message = _failure_message(state)
        return {
            "answer": message,
            "status": "degraded",
            "messages": [AIMessage(content=message)],
        }

    results = "\n\n".join(
        f"### {key}\n{frame.to_markdown()}" for key, frame in frames.items()
    )

    # A step with no frame did not return "no data" — its query never
    # succeeded. Letting the model infer emptiness from a missing table is how
    # you get "there were no sales" when the truth is "the query failed".
    missing = [step for step in state.get("plan", []) if step.id not in frames]
    if missing:
        results += "\n\n### Could not be retrieved\n" + "\n".join(
            f"- {step.question}" for step in missing
        )
        results += (
            "\n\nThese queries failed. Do not describe them as zero or as having "
            "no data. Say plainly that you could not retrieve them."
        )

    prompt = SYNTHESIS_PROMPT.format(
        persona=active_body(deps.personas),
        style=style_instruction(
            preferred(deps.preferences, state.get("user_id", ""))
        ),
        safety=SAFETY_RULES,
        question=question,
        results=results,
    )
    reply = deps.llm.invoke([HumanMessage(content=prompt)])
    scanned = scan_text(message_text(reply))

    return {
        "answer": scanned.text,
        "status": "degraded" if missing else state.get("status", "ok"),
        "messages": [AIMessage(content=scanned.text)],
    }


def _failure_message(state: TurnState) -> str:
    attempts = state.get("sql_attempts", [])
    failed = [a for a in attempts if a.failed]

    if not failed:
        return (
            "I couldn't find any data for that question. "
            "Could you tell me the time period and the metric you have in mind?"
        )

    last = failed[-1]
    problem = "; ".join(last.violations) if last.violations else (last.error or "")
    return (
        "I couldn't build a working query for that.\n\n"
        f"Last attempt:\n```sql\n{last.sql}\n```\n\n"
        f"What went wrong: {problem}\n\n"
        "Can you rephrase, or tell me which tables you expect this to come from?"
    )
