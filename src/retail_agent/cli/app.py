"""CLI entry point.

Deliberately thin. `main` imports argparse and the settings and nothing else,
so `retail-agent migrate` starts immediately instead of loading BigQuery,
pandas and langchain first — over a second of imports for a command that only
talks to Postgres.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from retail_agent.config import get_settings
from retail_agent.store.db import run_migrations


def main() -> int:
    parser = argparse.ArgumentParser(prog="retail-agent")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="start the interactive session")
    chat.add_argument("--user", default="exec", help="who is asking (for preferences)")
    chat.add_argument("--verbose", action="store_true", help="show debug logging")

    sub.add_parser("migrate", help="apply database migrations")

    evals = sub.add_parser("eval", help="score the agent's answers against reference SQL")
    evals.add_argument("--case", action="append", help="run only these case ids")
    evals.add_argument("--limit", type=int, help="run only the first N cases")
    evals.add_argument(
        "--threshold", type=float, default=None, help="minimum accuracy to pass"
    )
    evals.add_argument(
        "--baseline",
        help="a previous --json report; blocks on a regression against its accuracy",
    )
    evals.add_argument("--json", dest="json_path", help="write the full report here")

    compare = sub.add_parser("compare", help="put two --json eval reports side by side")
    compare.add_argument("left", help="a --json report, conventionally the older run")
    compare.add_argument("right", help="a --json report, conventionally the newer run")
    compare.add_argument("--left-name", dest="left_name", default="before")
    compare.add_argument("--right-name", dest="right_name", default="after")

    args = parser.parse_args()
    if args.command == "migrate":
        return _migrate()
    if args.command == "eval":
        from retail_agent.cli.evals import run_evals

        return run_evals(args)
    if args.command == "compare":
        from retail_agent.cli.evals import run_compare

        return run_compare(args)
    if args.command == "chat":
        # Imported here, not at module scope: this is the line that costs a
        # second, and only this subcommand needs it.
        from retail_agent.cli.chat import run_chat

        return run_chat(args)

    parser.print_help()
    return 1


def _migrate() -> int:
    console = Console()
    settings = get_settings()
    try:
        revision = run_migrations(settings.database_url)
    except Exception as err:
        render_error(
            console,
            f"Could not reach the database at {settings.database_url}.\n"
            f"Is it running? Try: docker compose up -d postgres\n\n{err}",
        )
        return 1

    console.print(f"Database is at revision [bold]{revision}[/bold].")
    return 0


def render_error(console: Console, message: str, turn_id: str = "") -> None:
    """Local copy so `migrate` does not import the render module's dependencies."""
    from rich.panel import Panel

    suffix = f"\n\n[dim]turn {turn_id}[/dim]" if turn_id else ""
    console.print(Panel(f"{message}{suffix}", title="Something went wrong", style="red"))


if __name__ == "__main__":
    sys.exit(main())
