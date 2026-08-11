"""The two subagents, offered to the supervisor as tools.

`langchain` 1.3 has no subagent middleware, so a subagent here is a compiled
`create_agent` wrapped in a `@tool`. That is the whole pattern, and it is how
every future capability arrives: a chart builder, a mailer, a web search each
become one of these without the supervisor changing shape.

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

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.middleware import analyst_middleware
from retail_agent.agent.prompts import (
    ANALYST_PROMPT,
    PERSONA_DEFAULT,
    REPORT_QA_PROMPT,
    REPORT_WRITER_PROMPT,
    SAFETY_RULES,
)
from retail_agent.agent.schema import render_schema_for_sql
from retail_agent.agent.tools import build_analyst_tools, recall
from retail_agent.knowledge.trios import (
    assumption_note,
    definitions_block,
    live_trios,
    sql_assumption_note,
    style_examples,
)
from retail_agent.llm.messages import message_text
from retail_agent.llm.resilience import resilient_call
from retail_agent.safety.egress import scan_text
from retail_agent.store.definitions import all_definitions, personal_definitions_block
from retail_agent.store.personas import active_body
from retail_agent.store.preferences import notes_for, preference_block

log = logging.getLogger(__name__)


def build_subagents(deps: AgentDeps, capture: TurnCapture) -> list[BaseTool]:
    """The subagent tools, bound to one turn.

    Built per turn rather than per process because the definitions, the persona
    and the user's preferences are all read at build time and all of them can
    change between turns. The graph had the opposite problem: its schema was
    bound when the agent was constructed, so a long-lived server never noticed
    a column being added.
    """

    @tool
    def analyst(question: str, runtime: ToolRuntime) -> str:
        """Query the retail data and report what it found.

        Pass the executive's question in full, keeping every business term
        exactly as they wrote it. If the question turns on a term nobody has
        defined, call `ask_for_definitions` before this, not after — a query
        written against a guess has already been paid for.
        """
        with capture.step("analyst") as step:
            found = recall(deps, question)
            capture.record_definitions([trio.id for trio in found])

            # Everything this executive has ever defined, not a lookup of terms
            # picked out of the question. Nothing picks terms out of a question
            # any more, and the whole set costs one read.
            known = all_definitions(deps.definitions, runtime.context.user_id)
            # Written by `ask_for_definitions` when nobody was there to answer.
            # Read rather than recomputed: this subagent no longer decides what
            # is unsettled, so the only honest source is what actually happened
            # earlier in the turn.
            assumed = list(capture.assumed_terms)
            step.detail = _describe(found, known, assumed)

            agent = create_agent(
                model=deps.llm,
                tools=build_analyst_tools(deps, capture),
                system_prompt=ANALYST_PROMPT.format(
                    definitions=_definitions(found, known, assumed),
                    schema=render_schema_for_sql(deps),
                    dataset=deps.settings.bq_dataset,
                ),
                middleware=analyst_middleware(deps),
            )
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=runtime.config,
            )

            answer = final_text(result)
            if assumed:
                # The disclosure travels with the findings so the supervisor
                # cannot report the number without it.
                answer += f"\n\n{assumption_note(assumed)}"
            return answer or "I could not produce an answer to that."

    @tool
    def report_writer(
        brief: str, title: str, runtime: ToolRuntime, show_to_executive: bool = True
    ) -> str:
        """Write a report from findings, save it, and show it to the executive.

        Pass everything the analyst told you, including the figures. This tool
        cannot query anything, so a number missing from the brief cannot appear
        in the report. `title` is what the executive will see in their library.

        The executive is shown the report itself, so your answer is one
        covering sentence — do not repeat the report. Set `show_to_executive`
        false only for a draft you are about to rework.
        """
        with capture.step("report_writer") as step:
            # Built once, outside the lambda `resilient_call` may invoke more
            # than once: `report_writer_system_prompt` reads the persona
            # store, the preference store and the trio corpus, and a retried
            # attempt should resend the same prompt rather than re-read all
            # three on every attempt.
            system_prompt = report_writer_system_prompt(deps, capture)
            # A plain call, not an agent: with no tools there is no loop for a
            # graph to run, and `resilient_call` supplies the retry and
            # fallback that `create_agent`'s middleware supplied to the others.
            reply = resilient_call(
                deps,
                lambda model: model.invoke(
                    [
                        SystemMessage(system_prompt),
                        HumanMessage(brief),
                    ]
                ),
            )

            # The last sweep before the text is shown or stored, and it happens
            # here rather than at save time because those are now the same
            # moment for one copy of the text. A report is read long after the
            # conversation that produced it, by people who were not in it.
            scanned = scan_text(message_text(reply))
            report = deps.reports.save(
                owner_id=runtime.context.user_id,
                session_id=runtime.context.session_id,
                title=title or "Untitled report",
                body=scanned.text,
            )
            capture.record_report(
                report.id, report.title, report.body, show=show_to_executive
            )

            step.detail = f"{len(report.body)} chars, saved {report.id}"
            shown = (
                "The executive has been shown it."
                if show_to_executive
                else "It was not shown to the executive."
            )
            return (
                f"Report {report.id} '{report.title}' written "
                f"({len(report.body)} chars).\n{shown}"
            )

    @tool
    def ask_about_report(report_id: str, question: str, runtime: ToolRuntime) -> str:
        """Answer a question about a report the executive has saved.

        Use this for anything about an existing report — what it concluded,
        what its action items were, what it says about a region. Get the id
        from `list_reports`. Report text is not kept in this conversation, so
        this is the only way to read one.
        """
        with capture.step("ask_about_report") as step:
            # Owner-scoped by the store's own query, so an id guessed or
            # carried over from another session reads nothing.
            report = deps.reports.get(
                owner_id=runtime.context.user_id, report_id=report_id
            )
            if report is None:
                step.detail = f"no report {report_id}"
                return (
                    f"No report {report_id} in your library. Use list_reports "
                    f"to see what is saved."
                )

            step.detail = f"{report.id}, {len(report.body)} chars"
            reply = resilient_call(
                deps,
                lambda model: model.invoke(
                    [
                        SystemMessage(
                            REPORT_QA_PROMPT.format(
                                persona=active_body(deps.personas) or PERSONA_DEFAULT,
                                safety=SAFETY_RULES,
                                title=report.title,
                                report=report.body,
                            )
                        ),
                        HumanMessage(question),
                    ]
                ),
            )
            return message_text(reply) or "I could not find that in the report."

    return [analyst, report_writer, ask_about_report]


def report_writer_system_prompt(deps: AgentDeps, capture: TurnCapture) -> str:
    """The report writer's system prompt for one call.

    A plain function rather than only a literal inside the tool closure, for
    the same reason `supervisor_system_prompt` exists in `middleware.py`: so a
    test can ask what reaches the model without building a model call.

    `.strip()` on the result, not just on the template — `REPORT_WRITER_PROMPT`
    ends `{examples}\n\n{style}`, and `style_examples` and `preference_block`
    both return `""` when there is nothing to say. Stripping the template
    trims the literal, but `.format()` runs after that, so a user with no
    notes and a question no trio covers would still get the blank lines the
    empty slots leave behind welded onto the end of the prompt.
    """
    # The other half of what a trio carries. `metric_definitions` says what to
    # measure and reaches the analyst; `report` shows how analysts here write a
    # finding — split by cohort, compare against a baseline, close with
    # numbered actions — and that is a property of the writing, so it belongs
    # here rather than in the SQL loop.
    consulted = [trio for trio in live_trios(deps.trios) if trio.id in capture.trio_ids]
    return REPORT_WRITER_PROMPT.format(
        persona=active_body(deps.personas) or PERSONA_DEFAULT,
        safety=SAFETY_RULES,
        examples=style_examples(consulted),
        style=preference_block(notes_for(deps.preferences, capture.user_id)),
    ).strip()


def _definitions(found: list, known: dict[str, str], assumed: list[str]) -> str:
    """Everything settled about this question's terms.

    Where both cover a term the executive's own definition is the one in
    force — `remember_definition` promised "from now on" — so the corpus block
    withholds it rather than render a second meaning the model could prefer.
    `sql_assumption_note` covers the rest — without it a model given no
    threshold reaches for a bind parameter, which nothing binds and BigQuery
    rejects.
    """
    blocks = [
        definitions_block(found, except_for=known),
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
