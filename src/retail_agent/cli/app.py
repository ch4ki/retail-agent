"""CLI entry point: `retail-agent chat` and `retail-agent migrate`."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from contextlib import contextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from rich.console import Console

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.graph import build_graph, run_turn
from retail_agent.cli.render import render_answer, render_banner, render_error
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.errors import describe_llm_error
from retail_agent.llm.provider import MissingCredentialsError, build_llm
from retail_agent.obs.tracing import configure_tracing
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.db import run_migrations

HELP = """
[bold]Commands[/bold]
  /help    show this
  /quit    exit

Everything else is treated as a question about the data.
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(prog="retail-agent")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="start the interactive session")
    chat.add_argument("--user", default="exec", help="who is asking (for preferences)")
    chat.add_argument("--verbose", action="store_true", help="show debug logging")

    sub.add_parser("migrate", help="apply database migrations")

    args = parser.parse_args()
    if args.command == "migrate":
        return _migrate()
    if args.command == "chat":
        return _chat(args)

    parser.print_help()
    return 1


def _migrate() -> int:
    console = Console()
    settings = get_settings()
    try:
        applied = run_migrations(settings.database_url)
    except Exception as err:
        render_error(
            console,
            f"Could not reach the database at {settings.database_url}.\n"
            f"Is it running? Try: docker compose up -d postgres\n\n{err}",
        )
        return 1

    console.print(f"Applied: {applied or 'nothing new'}")
    return 0


def _chat(args) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    console = Console()
    settings = get_settings()

    # Must happen before build_llm: the tracer reads os.environ when the model
    # is constructed, not when it is called.
    tracing = configure_tracing(settings)

    try:
        deps = AgentDeps(
            settings=settings,
            llm=build_llm(settings),
            source=BigQuerySource(settings),
            policy=PiiPolicy.default(),
        )
    except MissingCredentialsError as err:
        render_error(console, str(err))
        return 1
    except Exception as err:
        render_error(
            console,
            "Could not connect to BigQuery.\n"
            "Set GOOGLE_CLOUD_PROJECT in .env and run "
            "`gcloud auth application-default login`.\n\n"
            f"{err}",
        )
        return 1

    session_id = uuid.uuid4().hex[:8]
    render_banner(
        console,
        settings.llm_provider,
        settings.resolved_model,
        settings.google_cloud_project or "no project set",
        tracing_project=settings.langsmith_project if tracing else None,
    )

    with _checkpointer(console, settings.database_url) as saver:
        graph = build_graph(deps, checkpointer=saver)
        return _repl(console, graph, deps, args.user, session_id)


@contextmanager
def _checkpointer(console: Console, database_url: str):
    """Durable conversation state, degrading to in-memory if Postgres is down.

    A missing database should cost you history across restarts, not the
    ability to use the agent.
    """
    try:
        with PostgresSaver.from_conn_string(database_url) as saver:
            saver.setup()
            yield saver
            return
    except Exception as err:
        # Full detail only under --verbose; the console gets one actionable line.
        logging.getLogger(__name__).debug("postgres unavailable: %s", err)
        console.print(
            "[yellow]Postgres unreachable — this session will not be saved. "
            "Run `docker compose up -d postgres` to enable history.[/yellow]"
        )
    yield MemorySaver()


def _repl(console, graph, deps, user, session_id) -> int:
    while True:
        try:
            question = console.input("\n[bold cyan]›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return 0

        if not question:
            continue
        if question in {"/quit", "/exit"}:
            console.print("Bye.")
            return 0
        if question == "/help":
            console.print(HELP)
            continue

        _answer(console, graph, deps, user, session_id, question)


def _answer(console, graph, deps, user, session_id, question) -> None:
    try:
        with console.status("thinking…"):
            state = run_turn(
                graph,
                user_id=user,
                session_id=session_id,
                question=question,
                repair_budget=deps.settings.repair_budget,
                config={"configurable": {"thread_id": session_id}},
            )
    except Exception as err:  # the REPL must survive anything
        # Full detail goes to the log; the user gets one actionable line.
        logging.getLogger(__name__).exception("turn failed")
        render_error(
            console, describe_llm_error(err, provider=deps.settings.llm_provider)
        )
        return

    render_answer(console, state)


if __name__ == "__main__":
    sys.exit(main())
