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
from retail_agent.cli.render import (
    render_answer,
    render_banner,
    render_error,
    render_manifest,
    render_metrics,
    render_persona,
    render_personas,
    render_preferences,
    render_stored_trace,
    render_trace,
)
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.llm.errors import describe_llm_error
from retail_agent.llm.provider import MissingCredentialsError, build_llm
from retail_agent.obs.traces import build_trace_store
from retail_agent.obs.traces import from_state as trace_from_state
from retail_agent.obs.tracing import configure_tracing
from retail_agent.safety.pii import PiiPolicy
from retail_agent.store.db import run_migrations
from retail_agent.store.learning import InMemorySignalStore
from retail_agent.store.personas import build_persona_store
from retail_agent.store.preferences import build_preference_store, preferred
from retail_agent.store.reports import build_report_store

HELP = """
[bold]Commands[/bold]
  /help    show this
  /reports list saved reports
  /undo    restore the last deletion
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

    console.print(f"Database is at revision [bold]{applied}[/bold].")
    return 0


def build_deps(settings, *, llm, source, console=None) -> AgentDeps:
    """Assemble everything the graph needs from the stores.

    Split out of `_chat` so it can be exercised without credentials — a missing
    name in here used to surface only when a user ran the app, and was reported
    as a BigQuery problem.
    """

    def warn(message: str):
        return (lambda: console.print(f"[yellow]{message}[/yellow]")) if console else None

    return AgentDeps(
        settings=settings,
        llm=llm,
        source=source,
        policy=PiiPolicy.default(),
        traces=build_trace_store(settings),
        personas=build_persona_store(settings),
        preferences=build_preference_store(settings),
        signals=InMemorySignalStore(),
        reports=build_report_store(
            settings,
            on_degraded=warn(
                "Reports will not be saved — Postgres is unreachable. Run "
                "`docker compose up -d postgres && uv run retail-agent migrate`."
            ),
        ),
    )


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
        llm = build_llm(settings)
    except MissingCredentialsError as err:
        render_error(console, str(err))
        return 1

    try:
        source = BigQuerySource(settings)
    except Exception as err:
        render_error(
            console,
            "Could not connect to BigQuery.\n"
            "Set GOOGLE_CLOUD_PROJECT in .env and run "
            "`gcloud auth application-default login`.\n\n"
            f"{err}",
        )
        return 1

    try:
        deps = build_deps(settings, llm=llm, source=source, console=console)
    except Exception as err:
        # Anything here is a wiring fault, not the user's environment. Saying
        # "check your GCP project" would send them to fix something that is
        # not broken.
        logging.getLogger(__name__).exception("startup failed")
        render_error(console, f"The agent could not start: {err}")
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
    last_turn: dict = {}
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
        if question == "/reports":
            _show_reports(console, deps, user)
            continue
        if question == "/undo":
            _undo(console, deps, user)
            continue
        if question.startswith("/trace"):
            _trace(console, deps, user, last_turn, question)
            continue
        if question.startswith("/prefs"):
            _prefs(console, deps, user, question)
            continue
        if question.startswith("/persona"):
            _persona(console, deps, user, question)
            continue
        if question == "/metrics":
            render_metrics(console, deps.traces.metrics(owner_id=user))
            continue

        _learn(deps, user, question)
        last_turn = (
            _answer(console, graph, deps, user, session_id, question) or last_turn
        )
        _offer_proposal(console, deps, user)


def _prefs(console, deps, user, command) -> None:
    """`/prefs` shows them; `/prefs <setting> <value>` changes one;
    `/prefs accept|decline` answers a suggestion the agent made."""
    from retail_agent.store.preferences import PreferenceError, coerce, preferred

    parts = command.split()
    if len(parts) == 2 and parts[1] in {"accept", "decline"}:
        _answer_proposal(console, deps, user, accepted=parts[1] == "accept")
        return
    if len(parts) == 1:
        render_preferences(console, preferred(deps.preferences, user))
        return
    if len(parts) < 3:
        console.print("Usage: /prefs <setting> <value>  — /prefs lists them.")
        return

    field, value = parts[1], " ".join(parts[2:])
    try:
        parsed = coerce(field, value)
    except PreferenceError as err:
        console.print(f"[yellow]{err}[/yellow]")
        return

    updated = deps.preferences.set(user_id=user, **{field: parsed})
    console.print(f"Set [bold]{field}[/bold] to {value}.")
    render_preferences(console, updated)


def _answer_proposal(console, deps, user, *, accepted: bool) -> None:
    """Apply or refuse the pending suggestion.

    Either way the evidence for that field is cleared, so the counters that
    produced the question cannot immediately produce it again.
    """
    from retail_agent.store.learning import next_proposal
    from retail_agent.store.preferences import preferred

    proposal = next_proposal(
        deps.signals, user_id=user, current=preferred(deps.preferences, user)
    )
    if proposal is None:
        console.print("Nothing to accept or decline.")
        return

    if accepted:
        deps.preferences.set(user_id=user, **{proposal.field: proposal.value})
        console.print(f"Set [bold]{proposal.field}[/bold] to {proposal.value}.")
    else:
        deps.signals.decline(
            user_id=user, field=proposal.field, value=proposal.value
        )
        console.print("Left as it was.")
    deps.signals.clear(user_id=user, field=proposal.field)


def _offer_proposal(console, deps, user) -> None:
    """Ask, once there is enough evidence. Never change anything unasked: a
    personalisation the reader cannot account for is worse than none."""
    from retail_agent.store.learning import next_proposal
    from retail_agent.store.preferences import preferred

    proposal = next_proposal(
        deps.signals, user_id=user, current=preferred(deps.preferences, user)
    )
    if proposal is not None:
        console.print(f"\n[dim]{proposal.question()}[/dim]")


def _learn(deps, user, question) -> None:
    """Accumulate evidence from how the question was phrased. Deterministic, so
    the proposal can quote what the user actually typed."""
    from retail_agent.store.learning import detect

    try:
        for signal in detect(question):
            deps.signals.record(user_id=user, signal=signal)
    except Exception as err:  # learning is never worth a failed turn
        logging.getLogger(__name__).debug("signal not recorded: %s", err)


def _persona(console, deps, user, command) -> None:
    """`/persona`, `/persona show`, `/persona activate <name> [version]`.

    Activating by an older version is the rollback path: editing appends a
    version rather than overwriting, so the previous body is still there.
    """
    parts = command.split()
    action = parts[1] if len(parts) > 1 else "list"

    if action == "show":
        render_persona(console, deps.personas.active())
        return

    if action == "activate":
        if len(parts) < 3:
            console.print("Usage: /persona activate <name> [version]")
            return
        version = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
        try:
            persona = deps.personas.activate(name=parts[2], version=version)
        except KeyError:
            console.print(f"No persona named {parts[2]!r}. Try /persona list.")
            return
        console.print(f"Now speaking as [bold]{persona.name}[/bold] v{persona.version}.")
        return

    render_personas(console, deps.personas.list_personas(), deps.personas.active())


def _show_reports(console, deps, user) -> None:
    saved = deps.reports.list_reports(owner_id=user)
    if not saved:
        console.print("No saved reports yet.")
        return
    for report in saved:
        console.print(
            f"[bold]{report.title}[/bold]  "
            f"[dim]{report.id} · {report.created_at:%Y-%m-%d}[/dim]"
        )


def _undo(console, deps, user) -> None:
    restored = deps.reports.undo(owner_id=user)
    console.print(
        f"Restored {restored} report(s)." if restored else "Nothing to undo."
    )


def _answer(console, graph, deps, user, session_id, question) -> dict | None:
    """Run one turn. Returns the finished state so the REPL can `/trace` it."""
    config = {"configurable": {"thread_id": session_id}}
    state: dict = {}
    try:
        with console.status("thinking…"):
            state = run_turn(
                graph,
                user_id=user,
                session_id=session_id,
                question=question,
                config=config,
            )

        # The graph breaks before `await_confirmation` when a delete is staged.
        # Show the manifest, take the answer, and fill it in as though that node
        # had produced it.
        while graph.get_state(config).next == ("await_confirmation",):
            render_manifest(console, graph.get_state(config).values["pending_action"])
            typed = console.input("[bold yellow]›[/bold yellow] ").strip()
            graph.update_state(
                config, {"confirmation": typed}, as_node="await_confirmation"
            )
            with console.status("working…"):
                state = graph.invoke(None, config)
    except Exception as err:  # the REPL must survive anything
        # Full detail goes to the log; the user gets one actionable line.
        logging.getLogger(__name__).exception("turn failed")
        # The turn id makes a complaint a single lookup instead of an
        # investigation, so it goes on screen rather than only into the log.
        render_error(
            console,
            describe_llm_error(err, provider=deps.settings.llm_provider),
            turn_id=state.get("turn_id", ""),
        )
        return state or None

    _persist(deps, state)
    render_answer(console, state, prefs=preferred(deps.preferences, user))
    return state


def _persist(deps, state) -> None:
    """Record the finished turn. A trace is a debugging aid, so failing to write
    one must never cost the user their answer."""
    if not state.get("turn_id"):
        return
    try:
        deps.traces.record(trace_from_state(state))
    except Exception as err:
        logging.getLogger(__name__).debug("trace not recorded: %s", err)


def _trace(console, deps, user, last_turn, command) -> None:
    """`/trace` explains the last turn; `/trace <id>` reads one back from
    storage, which is what makes a user's complaint a single lookup."""
    _, _, wanted = command.partition(" ")
    wanted = wanted.strip()
    if not wanted:
        render_trace(console, last_turn)
        return

    stored = deps.traces.get(owner_id=user, turn_id=wanted)
    if stored is None:
        console.print(f"No trace for turn {wanted}.")
        return
    render_stored_trace(console, stored)


if __name__ == "__main__":
    sys.exit(main())
