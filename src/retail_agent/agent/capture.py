"""What one turn did, recorded on the way past.

The graph kept this in checkpointed state and read it back out with
`answer_from_state`. An agent has only a message list, and the rows it saw were
rendered into a `ToolMessage` string. Re-parsing that string to score or to
trace it would measure how the tool formatted its output — the same mistake
`evals/runner.py` documents about parsing prose.

So the tools write the real `MaskedFrame`, the executed SQL and the timings
here on the way past, and the eval and the trace read them back untouched.

One capture per turn, created by the caller and closed over by the tools. It is
deliberately not global: eval cases run sequentially today, but a shared capture
would silently attribute case 4's rows to case 3 the moment that changed.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from retail_agent.obs.traces import MAX_ANSWER_CHARS, TraceRecord
from retail_agent.safety.frame import MaskedFrame


@dataclass
class Step:
    """The one line of what a tool decided, for `/trace`."""

    detail: str = ""


@dataclass(frozen=True)
class PendingDelete:
    """A destructive operation resolved against the store, awaiting approval.

    Resolved by the interrupt's `when` predicate — read-only — and read back by
    the tool on the far side of the approval, so what the user was shown is
    exactly what gets deleted. Resolving twice would leave a gap, however small,
    between the manifest and the act.
    """

    action_id: str
    report_ids: tuple[str, ...]
    titles: tuple[str, ...]
    token: str  # what the user must type back, verbatim


@dataclass(frozen=True)
class PendingDefinition:
    """Terms the model asked about that this user has not already answered.

    Resolved by the interrupt's `when` predicate — read-only, and without a
    model call — and read back by the CLI, which asks about them in order. The
    same discipline as `PendingDelete`: what stops the turn is decided once, and
    what the user is then shown is exactly what stopped it.

    Which words are worth asking about is the model's judgement, made by calling
    `ask_for_definitions` at all. The filtering this holds is the other half:
    whichever of those words are already on file, dropped before anyone is
    interrupted, so a definition given last week is never asked for twice.
    """

    terms: tuple[str, ...]


@dataclass(frozen=True)
class WrittenReport:
    """A report this turn produced, held so the CLI can print it.

    The body is here rather than left to the model because the CLI prints this
    copy: it is the same text that was scanned and stored, so what the
    executive reads and what the library holds cannot drift apart. Everything
    the model receives about a report is an id and a headline.
    """

    report_id: str
    title: str
    body: str
    show: bool


@dataclass
class TurnCapture:
    """Mutable for the length of one turn, then read-only in practice.

    `question` is held because two things need it after the fact: the trace,
    and `note_preference`, which refuses to record a preference whose evidence
    the user never actually typed.
    """

    user_id: str = ""
    session_id: str = ""
    question: str = ""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    frame: MaskedFrame | None = None
    executed_sql: str = ""
    attempts: list[dict] = field(default_factory=list)
    events: list[tuple[str, int, str]] = field(default_factory=list)
    trio_ids: list[str] = field(default_factory=list)
    assumed_terms: list[str] = field(default_factory=list)
    redactions: int = 0
    calls: int = 0
    # `(action, note)` pairs the preference tools applied this turn, where
    # action is "added" or "removed". Held here so the CLI reports them itself
    # rather than trusting the model to mention it.
    preference_changes: list[tuple[str, str]] = field(default_factory=list)
    # Reports written this turn. The CLI prints those with `show`; the trace
    # keeps their ids so "what did this turn produce" is one lookup.
    reports_written: list[WrittenReport] = field(default_factory=list)
    # What this turn's thread will cost every later turn. Written by the
    # recorder middleware; a turn that died before it never has a figure.
    context_tokens: int = 0
    status: str = "ok"
    pending: PendingDelete | None = None
    pending_definition: PendingDefinition | None = None
    # The corpus retrieval for this turn's question, run once and read by both
    # sides of the definition interrupt. Only this half is cached: the trio
    # corpus cannot change mid-turn, but the personal definition store can —
    # the pause exists so the executive can write to it.
    recalled_trios: list | None = None

    @contextmanager
    def step(self, name: str) -> Iterator[Step]:
        """Time one tool call and file what it decided.

        Wrapping at the call site rather than inside each tool body means a tool
        added later is traced by virtue of being called, the way the graph's
        `_traced` wrapper made a new node traced by virtue of being registered.
        """
        started = time.perf_counter()
        step = Step()
        self.calls += 1
        try:
            yield step
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.events.append((name, elapsed_ms, step.detail))

    def record_attempt(
        self,
        sql: str,
        *,
        executed_sql: str = "",
        violations: Sequence[str] = (),
        error: str | None = None,
        frame: MaskedFrame | None = None,
        bytes_billed: int | None = None,
    ) -> None:
        """One try at a query, whether or not it ran.

        Failures are recorded too — `first_pass_validity` and
        `self_correction_rate` are both ratios over attempts, and a suite that
        only saw the successes would report a perfect first pass every time.

        `frame` and `executed_sql` are only updated on success, so a turn that
        fails the guard, repairs and then succeeds is scored on the query that
        ran rather than on the one that did not.
        """
        self.attempts.append(
            {
                "step_id": f"q{len(self.attempts) + 1}",
                "sql": sql,
                "executed_sql": executed_sql or None,
                "violations": list(violations),
                "error": error,
                "row_count": frame.row_count if frame else None,
                "bytes_billed": bytes_billed,
            }
        )
        if frame is not None:
            self.frame = frame
            self.executed_sql = executed_sql or sql
            self.redactions += frame.redactions

    def record_definitions(self, trio_ids: Sequence[str]) -> None:
        # Order-preserving union: the model may look the same term up twice, and
        # a trace should report what was consulted, not how often.
        for trio_id in trio_ids:
            if trio_id not in self.trio_ids:
                self.trio_ids.append(trio_id)

    def record_assumptions(self, terms: Sequence[str]) -> None:
        for term in terms:
            if term not in self.assumed_terms:
                self.assumed_terms.append(term)

    def record_report(
        self, report_id: str, title: str, body: str, *, show: bool
    ) -> None:
        self.reports_written.append(WrittenReport(report_id, title, body, show))

    @property
    def intent(self) -> str:
        """Derived rather than classified.

        The graph asked a model which of four things the turn was before doing
        any of them. Which tools actually ran is the same answer, arrived at
        after the fact and for free.
        """
        ran = {name for name, _, _ in self.events}
        if "analyst" in ran or "run_sql" in ran:
            return "analyze"
        if ran & {"report_writer", "ask_about_report", "list_reports", "delete_reports"}:
            return "report_op"
        if "describe_schema" in ran:
            return "schema"
        return "chat"

    def to_trace(self, answer: str) -> TraceRecord:
        """Everything already masked, and no row values.

        `frame` is the only place row values live and it is not among the
        fields, so a trace cannot become a second disclosure path.
        """
        return TraceRecord(
            turn_id=self.turn_id,
            session_id=self.session_id,
            owner_id=self.user_id,
            question=self.question,
            intent=self.intent,
            status=self.status,
            answer=answer[:MAX_ANSWER_CHARS],
            redactions=self.redactions,
            bytes_billed=sum(a.get("bytes_billed") or 0 for a in self.attempts),
            duration_ms=sum(duration for _, duration, _ in self.events),
            events=list(self.events),
            attempts=list(self.attempts),
            trios=list(self.trio_ids),
            assumptions=list(self.assumed_terms),
            preference_changes=list(self.preference_changes),
            report_ids=[report.report_id for report in self.reports_written],
            context_tokens=self.context_tokens,
        )
