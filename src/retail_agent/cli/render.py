"""Terminal presentation. Knows nothing about the agent's internals."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from retail_agent.agent.state import TurnState


def render_answer(console: Console, state: TurnState) -> None:
    answer = state.get("answer", "")
    if not answer:
        return

    console.print(Markdown(answer))

    footnotes = []
    if state.get("status") == "degraded":
        footnotes.append("partial answer — see the explanation above")
    redactions = state.get("redactions", 0)
    if redactions:
        footnotes.append(f"{redactions} personal-data values masked")
    attempts = len(state.get("sql_attempts", []))
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
