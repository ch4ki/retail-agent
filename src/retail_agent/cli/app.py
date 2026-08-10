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

    trios = sub.add_parser(
        "trios", help="compare the stored corpus against seeds.py"
    )
    trios.add_argument(
        "--force",
        action="store_true",
        help="overwrite the drifted trios from seeds.py",
    )

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

    args = parser.parse_args()
    if args.command == "migrate":
        return _migrate()
    if args.command == "trios":
        return run_trios(["--force"] if args.force else [])
    if args.command == "eval":
        from retail_agent.cli.evals import run_evals

        return run_evals(args)
    if args.command == "chat":
        # Imported here, not at module scope: this is the line that costs a
        # second, and only this subcommand needs it.
        from retail_agent.cli.chat import run_chat

        return run_chat(args)

    parser.print_help()
    return 1


def run_trios(argv, *, console: Console | None = None, store=None) -> int:
    """Report how the stored corpus differs from `seeds.py`, and optionally
    apply it.

    Two steps rather than one because the two cases are indistinguishable from
    here. `PostgresTrioStore.seed` inserts what is absent and leaves what is
    there, so an analyst's edit survives a restart — and a `seeds.py` edit
    never lands. A trio that differs is one or the other, and only the person
    running this knows which. So it prints the list and changes nothing until
    told, the same bargain the delete gate makes.

    `store` is injected so this is testable without a database; `main` passes
    nothing and gets the real one.
    """
    from retail_agent.knowledge.seeds import SEED_TRIOS
    from retail_agent.knowledge.trios import reseed, seed_drift

    console = console or Console()
    if store is None:
        try:
            from retail_agent.knowledge.trios import build_trio_store

            store = build_trio_store(get_settings())
        except Exception as err:
            render_error(console, f"Could not reach the corpus.\n\n{err}")
            return 1

    drift = seed_drift(store, SEED_TRIOS)
    if not drift:
        console.print(f"The corpus is up to date — {len(SEED_TRIOS)} trios.")
        return 0

    for trio_id, state in sorted(drift.items()):
        console.print(f"  [bold]{trio_id}[/bold] — {state}")

    if "--force" not in argv:
        console.print(
            f"\n{len(drift)} trio(s) differ from seeds.py. Nothing was changed.\n"
            "Run [bold]retail-agent trios --force[/bold] to overwrite them — "
            "which discards any edit made through the store."
        )
        return 0

    written = reseed(store, SEED_TRIOS)
    console.print(f"\nRewrote {len(written)} trio(s) from seeds.py.")
    return 0


def setup_checkpoint_tables(database_url: str) -> None:
    """Create LangGraph's checkpoint tables. Idempotent.

    Imported inside the function, not at module scope: `app.py` is deliberately
    thin so `retail-agent migrate` starts immediately, and `PostgresSaver`
    pulls in langgraph.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()


def _migrate(*, console: Console | None = None) -> int:
    console = console or Console()
    settings = get_settings()
    try:
        revision = run_migrations(settings.database_url)
        # Same command, same transaction boundary as every other table. A
        # migrate that upgraded the schema but left the checkpointer without
        # tables would exit 0 and fail on the next turn.
        setup_checkpoint_tables(settings.database_url)
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
