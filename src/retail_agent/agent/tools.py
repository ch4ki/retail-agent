"""The two tools that reach the warehouse and the corpus.

`run_sql` is the only path from BigQuery into model context anywhere in this
system, and masking sits inside it: the guard drops restricted columns so they
never leave the warehouse, and `mask_dataframe` runs before a single row is
rendered. That is where the PII guarantee lives now.

The graph made the same guarantee by ordering its edges. This one is stronger,
because it does not depend on the order anything runs in — there is no second
path to get a row out. `test_only_run_sql_reads_the_warehouse` is what keeps it
true as tools are added.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.state import attempt_record, frame_to_state, step_event
from retail_agent.knowledge.retrieval import retrieve
from retail_agent.knowledge.trios import (
    agreed_definitions,
    definitions_block,
    live_trios,
    lookup_definition,
)
from retail_agent.store.definitions import all_definitions
from retail_agent.safety.frame import MaskedFrame
from retail_agent.safety.pii import mask_dataframe
from retail_agent.safety.sql_guard import check_sql

log = logging.getLogger(__name__)

NOTHING_DEFINED = (
    "No agreed definition covers this question. Decide for yourself, and say "
    "in your answer which interpretation you used."
)

# What the graph spent a model call on. An empty result raises nothing — it is a
# successful query — so no retry middleware reacts to it, and `WHERE brand =
# 'Levis'` against a stored `Levi's` produces a confident zero. Saying so in the
# tool result costs nothing and reaches a model that is already in a loop.
EMPTY_HINT = (
    "\n\n_This returned nothing. That is usually an exact-match filter against a "
    "value stored differently (spelling, punctuation, case) rather than a real "
    "zero. Check the values that column actually holds, or match more loosely, "
    "before reporting this as zero._"
)


class GuardRejection(Exception):
    """The query never ran.

    Carries the violations so `ToolErrorMiddleware` can hand them back to the
    model, which is what the graph's repair edge did.
    """


def build_analyst_tools(deps: AgentDeps) -> list[BaseTool]:
    """Tools bound to one turn's `deps`.

    Closures rather than methods so each one can hold the turn's `deps`, and
    `@tool` rather than a bare function so the name, the argument schema and
    the description the model reads are derived from the definition itself —
    there is no second place to keep them in step.
    """

    @tool
    def run_sql(sql: str, runtime: ToolRuntime[Any, Any]) -> Command:
        """Run a read-only BigQuery query against theLook and return the rows.

        Write exactly one SELECT statement. Do not add a LIMIT; one is applied
        for you. If the query is rejected or fails you will be told why and may
        try again.
        """
        started = time.perf_counter()
        verdict = check_sql(
            sql,
            allowed_tables=deps.settings.allowed_tables,
            restricted_columns=deps.policy.restricted_columns(),
            default_limit=deps.settings.max_row_limit,
            max_limit=deps.settings.max_row_limit,
            qualify_with=deps.settings.bq_dataset,
        )
        if not verdict.ok:
            raise GuardRejection("; ".join(verdict.violations))

        deps.source.assert_within_budget(verdict.sql)
        result = deps.source.execute(verdict.sql)

        masked, report = mask_dataframe(
            result.rows, deps.policy, salt=deps.settings.pii_salt
        )
        frame = MaskedFrame.from_dataframe(
            masked,
            row_count=result.row_count,
            redactions=report.redactions,
            dropped_columns=report.dropped_columns,
            truncated=result.row_count > len(masked),
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=_render(frame),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "attempts": [
                    attempt_record(
                        sql=sql,
                        executed_sql=verdict.sql,
                        row_count=frame.row_count,
                        bytes_billed=result.bytes_billed,
                        index=len(runtime.state.get("attempts", [])),
                    )
                ],
                "events": [
                    step_event(
                        "run_sql",
                        started,
                        f"{frame.row_count} row(s), {result.bytes_billed} bytes",
                    )
                ],
                "frame": frame_to_state(frame),
                "executed_sql": verdict.sql,
                "redactions": report.redactions,
                "calls": 1,
            }
        )

    @tool
    def lookup_definitions(question: str, runtime: ToolRuntime[Any, Any]) -> Command:
        """Look up how the business defines the terms in a question.

        Use this when a term's meaning is a business decision rather than a
        column, and it is not already covered above. No list of examples on
        purpose: one used to be here, and it taught the model to recognise
        exactly those words and nothing else.
        """
        started = time.perf_counter()
        found = recall(deps, question)
        detail = (
            f"{len(found)} trio(s): {', '.join(t.id for t in found)}"
            if found
            else "no trio matched"
        )

        block = definitions_block(found)
        # An empty string reads to a model as a definition of nothing, and
        # it will proceed as though the term were settled. Saying so is the
        # whole protection the graph got from its undefined-term branch.
        content = block.strip() or NOTHING_DEFINED
        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=runtime.tool_call_id)
                ],
                "trio_ids": [trio.id for trio in found],
                "events": [step_event("lookup_definitions", started, detail)],
                "calls": 1,
            }
        )

    return [run_sql, lookup_definitions]


def recall(deps: AgentDeps, question: str) -> list:
    """The trios that settle this question. Never fails a turn.

    Retrieval is an improvement, not a dependency: an empty corpus is a valid
    state, and the undefined-term rule is what protects the answer when nothing
    comes back.
    """
    try:
        return retrieve(
            question,
            live_trios(deps.trios),
            dense_rank=deps.dense.rank if deps.dense is not None else None,
        )
    except Exception as err:
        log.warning("trio retrieval failed (%s); answering without it", err)
        return []


def settled_meanings(
    deps: AgentDeps, question: str, *, user_id: str, cache: dict
) -> dict[str, str]:
    """Every definition already in play this turn, as term → meaning.

    The agreed corpus for this question, then whatever this executive has
    settled themselves. The executive's own definition wins where both cover a
    term: `remember_definition` answers "I will use that from now on", and a
    trio merged later must not silently break that promise. The corpus fills
    gaps; it never overrides.

    The trios consulted are read back out of `cache["trios"]` by the caller —
    their meanings reach the model's context, so an answer that used them
    cannot show a trace that claims it used none.

    One function because `ask_for_definitions` calls it twice in the same
    turn — once before the pause, to decide whether there is anything left to
    ask, and again on replay after resume, to answer with whatever the
    executive just settled — and those two calls must agree on what already
    counts as known. They disagreed once, back when the check lived in two
    places instead of one: the CLI stopped to ask what "loyal" meant while the
    trio defining it sat in the same turn's retrieval, because each side had
    its own lookup.

    Retrieval runs once per turn, cached in `cache`, because it is the
    expensive half — with dense retrieval configured it is an embedding round
    trip — and its inputs are fixed for the turn. Two independent runs would
    also mean the two sides of the interrupt could, in principle, see two
    different corpora. The personal store is read fresh every call: the pause
    exists so the executive can write to it.

    `cache` is a plain dict the caller owns — `build_memory_tools` creates one
    per call, which is once per turn since every real caller rebuilds the
    agent per turn, so it lives exactly one turn and needs no identity of its
    own: it is never read after the turn ends and never shared between turns,
    and nothing in it is checkpointed, traced or scored.
    """
    if cache.get("trios") is None:
        cache["trios"] = recall(deps, question)
    found = cache["trios"]
    merged = dict(agreed_definitions(found))
    merged.update(all_definitions(deps.definitions, user_id))
    return merged


def partition_terms(
    known: dict[str, str], terms: Sequence[str | None]
) -> tuple[dict[str, str], list[str]]:
    """Split the asked-about terms into settled (term → meaning) and still open.

    One partition because `ask_for_definitions` calls it on both sides of its
    own interrupt and both calls must make the same call for every term — a
    stripping or matching rule applied in one and not the other is how they
    once came to disagree about "loyal".
    """
    settled: dict[str, str] = {}
    still_open: list[str] = []
    for term in terms:
        cleaned = term.strip() if term else ""
        if not cleaned:
            continue
        meaning = lookup_definition(known, cleaned)
        if meaning is None:
            still_open.append(cleaned)
        else:
            settled[cleaned] = meaning
    return settled, still_open


def _render(frame: MaskedFrame) -> str:
    """What the model sees of a result.

    `to_markdown` already caps the rows it prints and says how many were
    withheld, so a truncated result cannot read as a complete one.
    """
    body = frame.to_markdown()
    if frame.dropped_columns:
        body += f"\n\n_(withheld by policy: {', '.join(frame.dropped_columns)})_"
    if frame.truncated:
        body += (
            f"\n\n_This is a SAMPLE: {frame.row_count} rows matched and "
            f"{len(frame.rows)} are shown. The row count is exact — use it when "
            "the question asks how many. Do not total or average the rows shown "
            "and do not describe them as all of them._"
        )
    if frame.is_empty:
        body += EMPTY_HINT
    return body
