"""Terminal presentation. Knows nothing about the agent's internals."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from retail_agent.agent.state import TurnState


def render_answer(console: Console, state: TurnState, prefs=None) -> None:
    answer = state.get("answer", "")
    if not answer:
        return

    console.print(Markdown(answer))

    from retail_agent.store.preferences import DEFAULT_PREFERENCES

    prefs = prefs or DEFAULT_PREFERENCES
    if not prefs.show_attempt_footnote:
        return

    footnotes = []
    if state.get("status") == "degraded":
        footnotes.append("partial answer — see the explanation above")
    redactions = state.get("redactions", 0)
    if redactions:
        footnotes.append(f"{redactions} personal-data values masked")
    # A diagnosis marker is not a try the agent made; see SqlAttempt.
    attempts = len(
        [a for a in state.get("sql_attempts", []) if not a.is_diagnosis]
    )
    if attempts > 1:
        footnotes.append(f"{attempts} query attempts")

    if footnotes:
        console.print(f"[dim]{' · '.join(footnotes)}[/dim]")


def render_manifest(console: Console, action) -> None:
    """Show exactly what is about to be deleted, in full.

    Every title is printed. A truncated list would mean asking someone to
    confirm a deletion they cannot see.
    """
    lines = "\n".join(f"  • {title}" for title in action.titles)
    console.print(
        Panel(
            f"About to delete {len(action.report_ids)} report(s):\n\n{lines}\n\n"
            f"Type [bold]{action.token}[/bold] to confirm. Anything else cancels.",
            title="Confirm deletion",
            style="yellow",
        )
    )


def render_error(console: Console, message: str, turn_id: str = "") -> None:
    suffix = f"\n\n[dim]turn {turn_id}[/dim]" if turn_id else ""
    console.print(Panel(f"{message}{suffix}", title="Something went wrong", style="red"))


def render_banner(
    console: Console,
    provider: str,
    model: str,
    project: str,
    tracing_project: str | None = None,
) -> None:
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


def render_trace(console: Console, state) -> None:
    """The full message correspondence for a turn.

    Answers the question the footnote cannot: "3 query attempts" does not say
    whether that was a three-step plan or one step failing twice, nor what the
    guard objected to. This does.
    """
    events = (state or {}).get("events") or []
    if not events:
        console.print("No turn to trace yet — ask a question first.")
        return

    turn_id = state.get("turn_id", "unknown")
    total_ms = sum(event.duration_ms for event in events)
    console.print(
        f"[bold]turn {turn_id}[/bold]  [dim]intent={state.get('intent', '?')} · "
        f"status={state.get('status', '?')} · {total_ms} ms · "
        f"{state.get('redactions', 0)} masked[/dim]"
    )

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("node")
    table.add_column("ms", justify="right")
    table.add_column("what happened", overflow="fold")
    for event in events:
        table.add_row(event.node, str(event.duration_ms), event.detail)
    console.print(table)

    attempts = state.get("sql_attempts") or []
    if not attempts:
        return

    console.print("\n[bold]SQL attempts[/bold]")
    for index, attempt in enumerate(attempts, 1):
        outcome = (
            f"[red]rejected:[/red] {'; '.join(attempt.violations)}"
            if attempt.violations
            else f"[red]failed:[/red] {attempt.error}"
            if attempt.error
            else f"[green]{attempt.row_count} row(s), "
            f"{attempt.bytes_billed} bytes[/green]"
        )
        console.print(f"  [dim]{index}. {attempt.step_id}[/dim] {outcome}")
        console.print(f"     [dim]drafted:[/dim]  {attempt.sql}")
        if attempt.executed_sql and attempt.executed_sql != attempt.sql:
            console.print(f"     [dim]executed:[/dim] {attempt.executed_sql}")


def render_stored_trace(console: Console, trace) -> None:
    """A trace read back from storage.

    Same shape as the live one, but reconstructed from rows rather than from
    graph state — so `/trace <id>` answers a complaint about a turn that
    happened days ago in a session that has since ended.
    """
    console.print(
        f"[bold]turn {trace.turn_id}[/bold]  [dim]{trace.intent} · {trace.status} · "
        f"{trace.duration_ms} ms · {trace.redactions} masked · "
        f"{trace.bytes_billed} bytes[/dim]"
    )
    console.print(f"[dim]asked:[/dim] {trace.question}")

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("node")
    table.add_column("ms", justify="right")
    table.add_column("what happened", overflow="fold")
    for node, duration_ms, detail in trace.events:
        table.add_row(node, str(duration_ms), detail)
    console.print(table)

    if not trace.attempts:
        return
    console.print("\n[bold]SQL attempts[/bold]")
    for index, attempt in enumerate(trace.attempts, 1):
        if attempt.get("violations"):
            outcome = f"[red]rejected:[/red] {'; '.join(attempt['violations'])}"
        elif attempt.get("error"):
            outcome = f"[red]failed:[/red] {attempt['error']}"
        else:
            outcome = f"[green]{attempt.get('row_count')} row(s)[/green]"
        console.print(f"  [dim]{index}. {attempt.get('step_id')}[/dim] {outcome}")
        console.print(f"     [dim]drafted:[/dim]  {attempt.get('sql')}")
        executed = attempt.get("executed_sql")
        if executed and executed != attempt.get("sql"):
            console.print(f"     [dim]executed:[/dim] {executed}")


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
    table.add_row("degraded", f"{metrics['degraded_rate']:.0%}")
    table.add_row("SQL valid first pass", f"{metrics['first_pass_validity']:.0%}")
    table.add_row("self-correction succeeded", f"{metrics['self_correction_rate']:.0%}")
    table.add_row("personal-data values masked", str(metrics["redactions"]))
    table.add_row("bytes billed", f"{metrics['bytes_billed']:,}")
    console.print(f"[bold]Over the last {turns} turn(s)[/bold]")
    console.print(table)

    if metrics.get("node_p50_ms"):
        latency = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        latency.add_column("node")
        latency.add_column("p50 ms", justify="right")
        for node, p50 in metrics["node_p50_ms"].items():
            latency.add_row(node, str(p50))
        console.print("\n[bold]Median latency per node[/bold]")
        console.print(latency)


def render_personas(console: Console, personas, active) -> None:
    """The tone options and which one is live."""
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
        table.add_row(
            "→" if live else " ",
            f"[bold]{persona.name}[/bold]" if live else persona.name,
            str(persona.version),
            persona.updated_by,
        )
    console.print(table)


def render_persona(console: Console, persona) -> None:
    if persona is None:
        console.print("No persona is active; using the built-in default.")
        return
    console.print(
        Panel(
            persona.body,
            title=f"{persona.name} v{persona.version} · set by {persona.updated_by}",
            style="cyan",
        )
    )


def render_preferences(console: Console, prefs) -> None:
    """Current settings, with what each one does and how to change it."""
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
            table.add_row(trio.id if index == 0 else "", term, meaning)
    console.print(table)
    console.print(
        f"[dim]{len(live)} trio(s). A term with no entry here is stated as an "
        f"assumption in the answer.[/dim]"
    )


def render_definitions(console: Console, definitions) -> None:
    """What this user has told the agent, and how to change it.

    Shown separately from `/trios` on purpose: these are one person's working
    definitions, not decisions the analytics team agreed.
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
        table.add_row(entry.term, entry.definition)
    console.print(table)
    console.print("[dim]/definitions forget <term> to be asked again[/dim]")
