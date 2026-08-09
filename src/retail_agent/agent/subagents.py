"""The two subagents, offered to the supervisor as tools.

`langchain` 1.3 has no subagent middleware, so a subagent here is a compiled
`create_agent` wrapped in a plain callable. That is the whole pattern, and it is
how every future capability arrives: a chart builder, a mailer, a web search
each become one of these without the supervisor changing shape.

**analyst** owns the SQL loop. Its value is context isolation: a three-query
analysis produces several markdown tables of tool output that the supervisor
never needs and should not pay for on every subsequent model call.

**report_writer** has no tools on purpose. It exists for the prompt boundary —
persona, house structure and the user's format preference in one place, applied
to a brief rather than competing with orchestration for the supervisor's
attention.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain.agents import create_agent

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.middleware import analyst_middleware
from retail_agent.agent.prompts import (
    ANALYST_PROMPT,
    PERSONA_DEFAULT,
    REPORT_WRITER_PROMPT,
    SAFETY_RULES,
)
from retail_agent.agent.schema import render_schema_for_sql
from retail_agent.agent.tools import build_analyst_tools, recall
from retail_agent.knowledge.trios import (
    UNDEFINED_TERMS,
    assumption_note,
    definitions_block,
    live_trios,
    sql_assumption_note,
    style_examples,
    unresolved,
)
from retail_agent.llm.messages import message_text
from retail_agent.store.definitions import personal_definitions_block, remembered
from retail_agent.store.personas import active_body
from retail_agent.store.preferences import notes_for, preference_block

log = logging.getLogger(__name__)

# What the analyst returns instead of querying when a term is unsettled. The
# supervisor is told what to do with it by its own prompt; the property that
# matters — no spend before the term is settled — is enforced here, by
# returning before the subagent is built, rather than by asking a model nicely.
NEEDS_DEFINITION = (
    "I did not query anything. This question turns on **{term}**, and there is "
    "no agreed definition for it: {hint}. Ask the executive what it should "
    "mean, or call me again with assume_undefined=true to have me choose and "
    "say what I assumed."
)


def build_subagents(deps: AgentDeps, capture: TurnCapture) -> list[Callable]:
    """The subagent tools, bound to one turn.

    Built per turn rather than per process because the definitions, the persona
    and the user's preferences are all read at build time and all of them can
    change between turns. The graph had the opposite problem: its schema was
    bound when the agent was constructed, so a long-lived server never noticed
    a column being added.
    """

    def analyst(question: str, assume_undefined: bool = False) -> str:
        """Query the retail data and report what it found.

        Pass the executive's question in full, keeping every business term
        exactly as they wrote it. Set assume_undefined only after being told to
        decide for yourself about a term this tool said it needed defined.
        """
        with capture.step("analyst") as step:
            found = recall(deps, question)
            capture.record_definitions([trio.id for trio in found])

            open_terms = unresolved(question, found)
            known = remembered(deps.definitions, capture.user_id, open_terms)
            still_open = [term for term in open_terms if term not in known]

            # Only ask if the answer can be kept. Without somewhere to remember
            # it, the agent would ask the same person the same question every
            # turn — which is worse than assuming and saying so.
            if still_open and not assume_undefined and deps.definitions is not None:
                term = still_open[0]
                step.detail = f"needs a definition of {term}"
                return NEEDS_DEFINITION.format(
                    term=term, hint=UNDEFINED_TERMS.get(term, "it is undefined")
                )

            assumed = still_open if still_open else []
            capture.record_assumptions(assumed)
            step.detail = _describe(found, known, assumed)

            agent = create_agent(
                model=deps.llm,
                tools=build_analyst_tools(deps, capture),
                system_prompt=ANALYST_PROMPT.format(
                    definitions=_definitions(found, known, assumed),
                    schema=render_schema_for_sql(deps),
                    dataset=deps.settings.bq_dataset,
                ),
                middleware=analyst_middleware(deps.settings),
            )
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})

            answer = final_text(result)
            if assumed:
                # The disclosure travels with the findings so the supervisor
                # cannot report the number without it.
                answer += f"\n\n{assumption_note(assumed)}"
            return answer or "I could not produce an answer to that."

    def report_writer(brief: str) -> str:
        """Turn findings into a written report with action items.

        Pass everything the analyst told you, including the figures. This tool
        cannot query anything, so a number missing from the brief cannot appear
        in the report.
        """
        with capture.step("report_writer") as step:
            # The other half of what a trio carries. `metric_definitions` says
            # what to measure and reaches the analyst; `report` shows how
            # analysts here write a finding — split by cohort, compare against a
            # baseline, close with numbered actions — and that is a property of
            # the writing, so it belongs here rather than in the SQL loop.
            consulted = [
                trio for trio in live_trios(deps.trios) if trio.id in capture.trio_ids
            ]
            agent = create_agent(
                model=deps.llm,
                tools=[],
                system_prompt=REPORT_WRITER_PROMPT.format(
                    persona=active_body(deps.personas) or PERSONA_DEFAULT,
                    safety=SAFETY_RULES,
                    examples=style_examples(consulted),
                    style=preference_block(
                        notes_for(deps.preferences, capture.user_id)
                    ),
                ),
            )
            result = agent.invoke({"messages": [{"role": "user", "content": brief}]})
            body = final_text(result)
            step.detail = f"{len(body)} chars"
            return body

    return [analyst, report_writer]


def _definitions(found: list, known: dict[str, str], assumed: list[str]) -> str:
    """Everything settled about this question's terms, agreed first.

    Order matters: where both cover a term the corpus wins, so the model reads
    the reviewed decision before the personal one. `sql_assumption_note` covers
    the rest — without it a model given no threshold reaches for a bind
    parameter, which nothing binds and BigQuery rejects.
    """
    blocks = [
        definitions_block(found),
        personal_definitions_block(known),
        sql_assumption_note(assumed),
    ]
    return "\n\n".join(block for block in blocks if block) or (
        "No agreed definitions cover this question."
    )


def _describe(found: list, known: dict[str, str], assumed: list[str]) -> str:
    parts = [
        f"{len(found)} trio(s): {', '.join(t.id for t in found)}"
        if found
        else "no trio matched"
    ]
    if known:
        parts.append(f"user-defined: {', '.join(sorted(known))}")
    if assumed:
        parts.append(f"assuming: {', '.join(assumed)}")
    return "; ".join(parts)


def final_text(result: dict) -> str:
    """The closing message of a finished run.

    The last message with actual content: a run that ends on a tool call, or on
    an empty assistant turn, would otherwise report an empty answer and the
    egress scan would have nothing to look at.
    """
    for message in reversed(result.get("messages", [])):
        text = message_text(message).strip()
        if text:
            return text
    return ""
