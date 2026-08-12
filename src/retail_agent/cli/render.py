"""Terminal presentation. Knows nothing about the agent's internals."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

# Every value below that comes from the model or from a store (a report
# title, a trace answer, a persona body, a typed definition, a preference
# note...) goes through `escape()` before it reaches an f-string that also
# carries our own markup tags — Rich parses `[...]` in a plain string
# regardless of whether it looks intentional, in a `console.print` call, a
# `Panel` body, or a `Table` cell alike. Unescaped, a stray `[/bold]` in text
# nobody but the model or a store ever wrote doesn't just print wrong: Rich's
# parser raises `MarkupError` on it, and that can kill a turn (or, from
# `render_trace`, the session permanently — the string is persisted, so
# `/trace <id>` re-raises every time it's read back). Values that are ours —
# counts, ids we generate, node names, status enums — are left as real markup
# so their styling still works.


def render_answer(console: Console, answer: str, state=None, prefs=None) -> None:
    """The answer, plus what it cost to produce it.

    The footnote is the only place a masked value or a repaired query is
    admitted to in the normal flow. `/trace` has the detail; this says whether
    there is any detail worth asking for.

    `state` is the turn's checkpointed `TurnState` (or the dict `agent.invoke`
    returned) — the footnote reads `state["redactions"]` and
    `len(state["attempts"])` straight off it.

    Split into two calls below: the streaming CLI path prints the answer live,
    token by token, as it arrives — calling this afterward would print the
    same text a second time through `Markdown`. It calls `render_footnote`
    directly instead. This function still does both, unsplit, for anything
    that has the whole answer in hand up front and wants one call.
    """
    if not answer:
        return

    console.print(Markdown(answer))
    render_footnote(console, state, prefs)


def render_footnote(console: Console, state=None, prefs=None) -> None:
    """What the answer cost to produce — masked values, repaired queries.

    Pulled out of `render_answer` so a caller that already printed the answer
    itself (the streaming CLI path) can render just this part instead of
    printing the answer a second time to get it. `render_answer` calls this
    too, so the combined behaviour is unchanged for any caller that still
    wants it in one shot.
    """
    from retail_agent.store.preferences import DEFAULT_PREFERENCES

    prefs = prefs or DEFAULT_PREFERENCES
    if state is None or not prefs.show_attempt_footnote:
        return

    redactions = state.get("redactions", 0)
    attempts = state.get("attempts") or []

    footnotes = []
    if redactions:
        footnotes.append(f"{redactions} personal-data values masked")
    if len(attempts) > 1:
        footnotes.append(f"{len(attempts)} query attempts")

    if footnotes:
        console.print(f"[dim]{' · '.join(footnotes)}[/dim]")


def render_reports(console: Console, reports) -> None:
    """The reports this turn wrote, printed by the CLI rather than by the model.

    Each entry is a plain dict — `report_id`, `title`, `show`, and `body` the
    caller read back from the report store, since `TurnState` never carries a
    report's body (`reports.py`'s own comment: "the report store's copy is
    the one that gets read"). Reading the store's copy rather than trusting
    the model to retype it is what keeps what the executive reads and what the
    library holds from differing.

    The `show` filter is here rather than at the call site so that it is a
    behaviour with a test, instead of a condition a test would have to restate.
    """
    for report in reports:
        if not report["show"]:
            continue
        console.print()
        console.print(Markdown(report["body"]))
        # The title is the model's own text (`report_writer`'s `title` arg);
        # `report_id` is a uuid this process generated, never anyone else's
        # text, so it is left as-is.
        console.print(
            f"[dim]Saved as '{escape(report['title'])}' "
            f"(id {report['report_id']}) · /reports[/dim]"
        )


def render_confirmation(console: Console, description: str) -> None:
    """Show exactly what is about to be deleted, in full.

    Every title is printed — the manifest is built by `delete_reports` itself,
    which resolved the target set against the store before pausing. A
    truncated list would mean asking someone to confirm a deletion they cannot
    see.

    `description` is `render_manifest`'s text, which embeds every matched
    report's title — the model's own text, via `report_writer` — so it goes
    through `escape()` before it reaches the panel, the same as everywhere
    else a title is shown.
    """
    console.print(Panel(escape(description), title="Confirm deletion", style="yellow"))


def render_definition_prompt(
    console: Console, term: str, options: list[str]
) -> None:
    """The choice offered when a question turns on a term nobody has settled.

    The two fixed choices are numbered after the generated ones rather than at
    fixed positions, so their numbers move with the list instead of being
    something to memorise. Both are always present: the generated options can be
    empty — a model call is allowed to fail here — and a prompt with no way out
    would be a worse failure than the assumption it exists to prevent.

    `term` and every `option` are the model's own text — the term it asked
    about, and the candidate definitions it proposed — so both are escaped
    before they reach the console.
    """
    console.print(f"\n[bold]{escape(term)}[/bold] needs a definition\n")
    for index, option in enumerate(options, 1):
        console.print(f"  [bold cyan]{index}[/bold cyan]  {escape(option)}")
    console.print(f"  [bold cyan]{len(options) + 1}[/bold cyan]  [dim]something else — I'll type it[/dim]")
    console.print(
        f"  [bold cyan]{len(options) + 2}[/bold cyan]  "
        f"[dim]decide for me, and say what you assumed[/dim]"
    )
    console.print("\n[dim]Pick a number, or just type what it means. Empty to cancel.[/dim]")


def render_error(console: Console, message: str, turn_id: str = "") -> None:
    """The one panel every failed turn ends on.

    `message` can hold a provider's own error text (`describe_llm_error`
    embeds it verbatim after redaction) or, upstream of this function, a
    model-supplied string reaching it through some other path — neither is
    console markup, and an unmatched `[/bold]` in it is not just cosmetic:
    Rich's markup parser raises `MarkupError` on a stray closing tag, and
    that would escape `_answer`'s own exception handling because it happens
    *inside* the call that reports the turn already failed. `escape()` is
    the same guard `_vet_definition` uses on typed input for the same
    reason — the panel must not be able to raise from what it is showing.
    `turn_id` is ours, not the model's, so its `[dim]...[/dim]` wrapping is
    left as real markup.
    """
    suffix = f"\n\n[dim]turn {turn_id}[/dim]" if turn_id else ""
    console.print(
        Panel(f"{escape(message)}{suffix}", title="Something went wrong", style="red")
    )


def render_banner(
    console: Console,
    provider: str,
    model: str,
    project: str,
    tracing_project: str | None = None,
) -> None:
    """`provider`/`model`/`project`/`tracing_project` all come from `Settings`
    (env vars), not from the model or a mutable store — the operator who
    deploys this process controls them, not anyone talking to it — so left as
    real markup, unlike the values elsewhere in this file."""
    body = (
        "Ask about orders, products, customers or revenue.\n"
        "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit."
    )
    if tracing_project:
        # Prompts leave the machine when this is on, so say so.
        body += f"\n[dim]Tracing to LangSmith project '{tracing_project}'.[/dim]"

    console.print(
        Panel(body, title=f"Retail Analysis Agent · {provider}:{model} · {project}")
    )


def render_trace(console: Console, trace) -> None:
    """The full tool correspondence for a turn.

    Answers the question the footnote cannot: "3 query attempts" does not say
    whether that was three separate questions or one query failing twice, nor
    what the guard objected to. This does.

    One renderer for the live turn and for one read back from storage, because
    both are now a `TraceRecord` — the live one built by `trace_from_state(...)`.
    There used to be two near-identical renderers, which is how they drifted:
    only one of them showed bytes billed.

    `trace.question` is what the executive typed; `trace.answer` is the
    model's — both go through `escape()`. `intent`/`status` are fixed-vocabulary
    labels this codebase assigns, and everything else on the header line is a
    number, so those are left as real markup. This is also the sink the
    `/trace <id>` command reads back from storage — an unescaped answer here
    does not just fail once, it fails every time that stored turn is read,
    which is the shape the C3 regression actually took.
    """
    if trace is None:
        console.print("No turn to trace yet — ask a question first.")
        return

    console.print(
        f"[bold]turn {trace.turn_id}[/bold]  [dim]{trace.intent} · {trace.status} · "
        f"{trace.duration_ms} ms · {trace.redactions} masked · "
        f"{trace.bytes_billed} bytes · {trace.context_tokens} ctx tokens[/dim]"
    )
    console.print(f"[dim]asked:[/dim] {escape(trace.question)}")
    if trace.answer:
        console.print(f"[dim]answered:[/dim] {escape(trace.answer)}")
    _render_reasons(console, trace)
    _render_events(console, trace.events)
    _render_attempts(console, trace.attempts)


def _render_reasons(console: Console, trace) -> None:
    """What the turn was reasoning from.

    Above the steps rather than below them, because a disputed number is
    challenged on its definition far more often than on its timings. A term
    listed as assumed is the single most useful line in a trace: it says the
    agent chose, and what it chose about.

    `trace.trios` and `trace.report_ids` are ids this codebase mints — trio
    ids from the curated corpus, report ids this process generated — so they
    are left as real markup. `assumptions` are the model's own words about a
    term it chose not to ask about, and a preference's `value` is the note
    text `note_preference` recorded from the executive's own request
    (`"keep answers under three sentences"`), so both are escaped. `field` is
    one of a fixed, code-defined set of preference names.
    """
    if trace.trios:
        console.print(f"[dim]definitions used:[/dim] {', '.join(trace.trios)}")
    if trace.report_ids:
        # Where a deep dive into a disputed report starts. The body is in the
        # library, not here — this is the pointer to it.
        console.print(f"[dim]reports written:[/dim] {', '.join(trace.report_ids)}")
    if trace.assumptions:
        console.print(f"[yellow]assumed:[/yellow] {escape(', '.join(trace.assumptions))}")
    for field, value in trace.preference_changes:
        console.print(f"[dim]set as your default:[/dim] {field} = {escape(value)}")


def _render_events(console: Console, events) -> None:
    """One row per tool call, from `(step, duration_ms, detail)` triples.

    An empty list is a fact about the turn — the model answered from the
    conversation without reaching for anything — so it is stated. Printing the
    headers with nothing under them read as a renderer that had broken.

    `node` is a graph node name from this codebase's own fixed topology, left
    alone. `detail` is a free-text line a tool wrote about what it did, and
    several tools build it from a term or a value the model or the executive
    supplied (`ask_for_definitions`' `_describe_settled`, `note_preference`'s
    "remembered {term}"), so it is escaped — table cells parse markup on a
    plain string exactly like any other `console.print` target.
    """
    if not events:
        console.print("[dim]no tools were called — answered from the conversation[/dim]")
        return

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("step")
    table.add_column("ms", justify="right")
    table.add_column("what happened", overflow="fold")
    for node, duration_ms, detail in events:
        table.add_row(node, str(duration_ms), escape(detail or ""))
    console.print(table)


def _render_attempts(console: Console, attempts) -> None:
    """Every draft, what became of it, and the query the warehouse actually saw.

    `sql`/`executed_sql` are the model's own drafts. `violations` are the SQL
    guard's messages, but they quote the offending table/column name straight
    out of the model's SQL (`sql_guard._check_tables`/`_check_projections`),
    and `error` is the warehouse's own message, which can likewise echo back
    part of the query — so all four are escaped. `step_id` is one of this
    codebase's own fixed step labels ("draft", "repair-1", ...), not model
    text, and is left alone.
    """
    if not attempts:
        return

    console.print("\n[bold]SQL attempts[/bold]")
    for index, attempt in enumerate(attempts, 1):
        field = attempt.get
        violations, error = field("violations"), field("error")
        if violations:
            outcome = f"[red]rejected:[/red] {escape('; '.join(violations))}"
        elif error:
            outcome = f"[red]failed:[/red] {escape(error)}"
        else:
            billed = field("bytes_billed")
            billed_text = f", {billed} bytes" if billed else ""
            outcome = f"[green]{field('row_count')} row(s){billed_text}[/green]"

        console.print(f"  [dim]{index}. {field('step_id')}[/dim] {outcome}")
        console.print(f"     [dim]drafted:[/dim]  {escape(field('sql'))}")
        executed = field("executed_sql")
        if executed and executed != field("sql"):
            console.print(f"     [dim]executed:[/dim] {escape(executed)}")


def render_metrics(console: Console, metrics: dict) -> None:
    """Agent-level metrics over recent turns.

    Rates are shown as percentages with their denominator, because "50%
    self-correction" over two turns is not the same claim as over two hundred.
    """
    if not metrics.get("turns"):
        console.print("No turns recorded yet.")
        return

    turns = metrics["turns"]
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("metric", style="dim")
    table.add_column("value")
    table.add_row("turns", str(turns))
    table.add_row("SQL valid first pass", f"{metrics['first_pass_validity']:.0%}")
    table.add_row("self-correction succeeded", f"{metrics['self_correction_rate']:.0%}")
    table.add_row("personal-data values masked", str(metrics["redactions"]))
    table.add_row("bytes billed", f"{metrics['bytes_billed']:,}")
    table.add_row("context tokens (max)", f"{metrics['context_tokens_max']:,}")
    table.add_row("context tokens (median)", f"{metrics['context_tokens_p50']:,}")
    console.print(f"[bold]Over the last {turns} turn(s)[/bold]")
    console.print(table)

    if metrics.get("node_p50_ms"):
        latency = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        latency.add_column("step")
        latency.add_column("p50 ms", justify="right")
        for step, p50 in metrics["node_p50_ms"].items():
            latency.add_row(step, str(p50))
        console.print("\n[bold]Median latency per step[/bold]")
        console.print(latency)


def render_personas(console: Console, personas, active) -> None:
    """The tone options and which one is live.

    `name` and `updated_by` are store text — a persona is edited outside this
    CLI, and nothing validates what goes into either field — so both are
    escaped even where they carry real `[bold]` styling below.
    """
    if not personas:
        console.print("No personas saved.")
        return

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column(" ")
    table.add_column("name")
    table.add_column("v", justify="right")
    table.add_column("updated by", style="dim")
    for persona in personas:
        live = active is not None and persona.name == active.name
        name = escape(persona.name)
        table.add_row(
            "→" if live else " ",
            f"[bold]{name}[/bold]" if live else name,
            str(persona.version),
            escape(persona.updated_by),
        )
    console.print(table)


def render_persona(console: Console, persona) -> None:
    """`body` is the persona's full system-prompt text and `name`/`updated_by`
    land in the panel's `title=`, which Rich parses as markup independently of
    any `markup=` argument on the `console.print` call — so all three are
    escaped before they reach `Panel`, not just the body."""
    if persona is None:
        console.print("No persona is active; using the built-in default.")
        return
    console.print(
        Panel(
            escape(persona.body),
            title=(
                f"{escape(persona.name)} v{persona.version} · "
                f"set by {escape(persona.updated_by)}"
            ),
            style="cyan",
        )
    )


def render_preferences(console: Console, prefs, notes: Sequence[str] = ()) -> None:
    """Current settings and saved notes, with how to change either.

    The notes come first because they are what the agent actually applies now:
    one typed setting is left, and the rest of what an executive asks for lives
    in their own words. Omitting them made this screen actively misleading —
    saving a note ends with "/prefs to change it", and `/prefs` did not show it.

    Each `note` is `note_preference`'s recording of the executive's own
    request, in the model's paraphrase — store text, escaped before print.
    """
    if notes:
        console.print("[bold]You have asked for[/bold]")
        for note in notes:
            console.print(f"  • {escape(note)}")
        console.print(
            "[dim]Say 'forget that', or name one, to drop it.[/dim]\n"
        )

    from retail_agent.store.preferences import DESCRIPTIONS

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("setting")
    table.add_column("value")
    table.add_column("accepts", style="dim", overflow="fold")
    for field, description in DESCRIPTIONS.items():
        table.add_row(field, str(getattr(prefs, field)), description)
    console.print(table)
    console.print("[dim]Change one with: /prefs <setting> <value>[/dim]")


def render_trios(console: Console, trios) -> None:
    """The Golden Bucket, and what each entry settles.

    Shown as definitions rather than as questions, because the definition is
    what actually changes an answer.

    `trio.id` is the corpus's own identifier, left alone. `term`/`meaning` are
    the analyst-authored content of the corpus — free text, nothing here
    validates its shape — so both are escaped.
    """
    live = [t for t in trios if t.superseded_by is None]
    if not live:
        console.print("The Golden Bucket is empty — every business term will be assumed.")
        return

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("trio")
    table.add_column("defines")
    table.add_column("meaning", overflow="fold")
    for trio in live:
        for index, (term, meaning) in enumerate(sorted(trio.metric_definitions.items())):
            table.add_row(trio.id if index == 0 else "", escape(term), escape(meaning))
    console.print(table)
    console.print(
        f"[dim]{len(live)} trio(s). A term with no entry here is stated as an "
        f"assumption in the answer.[/dim]"
    )


def render_definitions(console: Console, definitions) -> None:
    """What this user has told the agent, and how to change it.

    Shown separately from `/trios` on purpose: these are one person's working
    definitions, not decisions the analytics team agreed.

    `entry.term` and `entry.definition` are typed by this executive (or
    accepted from a model-generated option) through `_vet_definition` — the
    same untrusted text `render_definition_prompt` shows while it is still
    being settled, so both are escaped here too.
    """
    if not definitions:
        console.print(
            "You have not defined any terms yet. I will ask when one comes up."
        )
        return

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("term")
    table.add_column("your definition", overflow="fold")
    for entry in definitions:
        table.add_row(escape(entry.term), escape(entry.definition))
    console.print(table)
    console.print("[dim]/definitions forget <term> to be asked again[/dim]")
