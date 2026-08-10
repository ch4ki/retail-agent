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
from collections.abc import Callable, Sequence

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.deps import AgentDeps
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


def build_analyst_tools(deps: AgentDeps, capture: TurnCapture) -> list[Callable]:
    """Tools bound to one turn's capture.

    Closures rather than methods so that `create_agent` receives plain callables
    with docstrings — which is the tool description the model actually reads.
    """

    def run_sql(sql: str) -> str:
        """Run a read-only BigQuery query against theLook and return the rows.

        Write exactly one SELECT statement. Do not add a LIMIT; one is applied
        for you. If the query is rejected or fails you will be told why and may
        try again.
        """
        with capture.step("run_sql") as step:
            verdict = check_sql(
                sql,
                allowed_tables=deps.settings.allowed_tables,
                restricted_columns=deps.policy.restricted_columns(),
                default_limit=deps.settings.max_row_limit,
                max_limit=deps.settings.max_row_limit,
                qualify_with=deps.settings.bq_dataset,
            )
            if not verdict.ok:
                capture.record_attempt(sql, violations=verdict.violations)
                step.detail = f"guard rejected — {'; '.join(verdict.violations)}"
                raise GuardRejection("; ".join(verdict.violations))

            try:
                deps.source.assert_within_budget(verdict.sql)
                result = deps.source.execute(verdict.sql)
            except Exception as err:
                capture.record_attempt(
                    sql, executed_sql=verdict.sql, error=str(err)
                )
                step.detail = str(err)
                raise

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
            capture.record_attempt(
                sql,
                executed_sql=verdict.sql,
                frame=frame,
                bytes_billed=result.bytes_billed,
            )
            step.detail = f"{frame.row_count} row(s), {result.bytes_billed} bytes"
            return _render(frame)

    def lookup_definitions(question: str) -> str:
        """Look up how the business defines the terms in a question.

        Use this when a term's meaning is a business decision rather than a
        column, and it is not already covered above. No list of examples on
        purpose: one used to be here, and it taught the model to recognise
        exactly those words and nothing else.
        """
        with capture.step("lookup_definitions") as step:
            found = recall(deps, question)
            capture.record_definitions([trio.id for trio in found])
            step.detail = (
                f"{len(found)} trio(s): {', '.join(t.id for t in found)}"
                if found
                else "no trio matched"
            )

            block = definitions_block(found)
            # An empty string reads to a model as a definition of nothing, and
            # it will proceed as though the term were settled. Saying so is the
            # whole protection the graph got from its undefined-term branch.
            return block.strip() or NOTHING_DEFINED

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


def settled_meanings(deps: AgentDeps, capture: TurnCapture) -> dict[str, str]:
    """Every definition already in play this turn, as term → meaning.

    The agreed corpus for this question, then whatever this executive has
    settled themselves. The executive's own definition wins where both cover a
    term: `remember_definition` answers "I will use that from now on", and a
    trio merged later must not silently break that promise. The corpus fills
    gaps; it never overrides.

    The trios consulted are recorded on the capture: their meanings reach the
    model's context, so an answer that used them cannot show a trace that
    claims it used none.

    One function because there are two places that must agree on whether a word
    needs asking about, and they run on opposite sides of an interrupt. The
    tool body decides it for headless callers; the gate's `when` predicate
    decides it for the CLI, *before* that body runs. They disagreed once — the
    CLI stopped to ask what "loyal" meant while the trio defining it sat in the
    same turn's retrieval — and the reason was that each had its own lookup.

    Retrieval runs once per turn, cached on the capture, because it is the
    expensive half — with dense retrieval configured it is an embedding round
    trip — and its inputs are fixed for the turn. Two independent runs would
    also mean the two sides of the interrupt could, in principle, see two
    different corpora. The personal store is read fresh every call: the pause
    exists so the executive can write to it.
    """
    if capture.recalled_trios is None:
        capture.recalled_trios = recall(deps, capture.question)
    found = capture.recalled_trios
    capture.record_definitions([trio.id for trio in found])
    merged = dict(agreed_definitions(found))
    merged.update(all_definitions(deps.definitions, capture.user_id))
    return merged


def partition_terms(
    known: dict[str, str], terms: Sequence[str | None]
) -> tuple[dict[str, str], list[str]]:
    """Split the asked-about terms into settled (term → meaning) and still open.

    One partition because the gate's `when` predicate and the tool body must
    make the same call for every term — a stripping or matching rule applied
    in one and not the other is how they came to disagree about "loyal".
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
