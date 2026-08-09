"""The interactive session.

Separate from `app.py` so `retail-agent migrate` does not pay for it. Importing
this module pulls in BigQuery, pandas and langchain — over a second before
anything runs — and a database migration needs none of that.

Every import here stays at module level. Deferring them into functions would
buy the same startup saving and reintroduce the failure it is meant to prevent:
a name that resolves everywhere except the one path a user takes.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from rich.console import Console

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.subagents import final_text
from retail_agent.agent.supervisor import build_agent
from retail_agent.bootstrap import build_deps
from retail_agent.cli.render import (
    render_answer,
    render_banner,
    render_confirmation,
    render_definitions,
    render_error,
    render_metrics,
    render_persona,
    render_personas,
    render_preferences,
    render_trace,
    render_trios,
)
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.knowledge.trios import live_trios
from retail_agent.llm.errors import describe_llm_error
from retail_agent.llm.provider import MissingCredentialsError, build_llm
from retail_agent.obs.tracing import configure_tracing
from retail_agent.store.preferences import preferred

HELP = """
[bold]Commands[/bold]
  /help    show this
  /reports list saved reports
  /undo    restore the last deletion
  /trace   explain the last turn; /trace <id> reads a stored one back
  /metrics first-pass SQL validity, self-correction, latency per step
  /trios   the analyst definitions the agent answers from
  /definitions what you told it terms mean; forget <term> to be asked again
  /prefs   answer format, depth, table size
  /persona list|show|activate <name> — change the tone, no restart
  /quit    exit (/exit works too)

Everything else is treated as a question about the data.
""".strip()


def run_chat(args) -> int:
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
        return _repl(console, deps, saver, args.user, session_id)


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


def _repl(console, deps, saver, user, session_id) -> int:
    last_trace = None
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
            _trace(console, deps, user, last_trace, question)
            continue
        if question.startswith("/prefs"):
            _prefs(console, deps, user, question)
            continue
        if question.startswith("/persona"):
            _persona(console, deps, user, question)
            continue
        if question.startswith("/definitions"):
            _definitions(console, deps, user, question)
            continue
        if question == "/trios":
            render_trios(console, live_trios(deps.trios))
            continue
        if question == "/metrics":
            render_metrics(console, deps.traces.metrics(owner_id=user))
            continue

        last_trace = (
            _answer(console, deps, saver, user, session_id, question) or last_trace
        )


def _prefs(console, deps, user, command) -> None:
    """`/prefs` shows them; `/prefs <setting> <value>` changes one."""
    from retail_agent.store.preferences import PreferenceError, coerce, preferred

    parts = command.split()
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


def _definitions(console, deps, user, command) -> None:
    """`/definitions` lists what you told the agent; `/definitions forget <term>`
    drops one so it asks again.

    There is no `promote`. Writing one person's definition into the corpus
    everyone answers from is a change to shared ground truth, and §5.1 puts a
    human review gate in front of that — a gate this prototype does not have.
    A command that skips it would contradict the design it is meant to
    demonstrate.
    """
    parts = command.split()
    if len(parts) >= 3 and parts[1] == "forget":
        term = " ".join(parts[2:])
        dropped = deps.definitions.forget(user_id=user, term=term)
        console.print(
            f"Forgot {term!r}; I will ask next time."
            if dropped
            else f"Nothing remembered for {term!r}."
        )
        return

    render_definitions(console, deps.definitions.list_definitions(user_id=user))


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


def _answer(console, deps, saver, user, session_id, question):
    """Run one turn. Returns its trace so the REPL can `/trace` it.

    A fresh agent and a fresh capture per turn: both close over the turn's
    identity, and the persona and preferences are read per model call, so
    rebuilding costs nothing and rules out a stale binding.
    """
    capture = TurnCapture(user_id=user, session_id=session_id, question=question)
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Inside the guard, not above it: assembling the agent reads the persona
        # store and the tool list, and a REPL that dies before its own error
        # handler is a REPL that dies.
        agent = build_agent(deps, capture, checkpointer=saver)
        with console.status("thinking…"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}, config
            )

        # A destructive tool call pauses the agent before it runs. The manifest
        # was resolved against the store by the approval gate, so what is shown
        # is what would go — and the typed token means a bulk delete cannot be
        # approved by reflex.
        while _pending(result):
            approved = _confirm(console, result, capture)
            with console.status("working…"):
                decision = {"type": "approve"} if approved else {
                    "type": "reject",
                    "message": "The executive did not confirm. Nothing was deleted.",
                }
                result = agent.invoke(
                    # A dict with `decisions`, not a bare list: the middleware
                    # subscripts the resume value by name, and a list resumes
                    # with a TypeError that surfaces as a failed turn.
                    Command(resume={"decisions": [decision]}),
                    config,
                )
    except Exception as err:  # the REPL must survive anything
        # Full detail goes to the log; the user gets one actionable line. The
        # turn id makes a complaint a single lookup rather than an
        # investigation, so it goes on screen and not only into the log.
        logging.getLogger(__name__).exception("turn failed")
        render_error(
            console,
            describe_llm_error(err, provider=deps.settings.llm_provider),
            turn_id=capture.turn_id,
        )
        return None

    answer = final_text(result)
    render_answer(console, answer, capture, prefs=preferred(deps.preferences, user))
    for field, value in capture.preference_changes:
        # Said by the CLI rather than left to the model: a setting that changed
        # without the reader being told is the failure this design cares about.
        console.print(f"[dim]Saved {field} = {value} as your default. /prefs to change it.[/dim]")
    # The trace was written by the recorder middleware, on every path out. This
    # is the same record, kept so `/trace` needs no round trip to storage.
    return capture.to_trace(answer)


def _pending(result) -> bool:
    return bool(result.get("__interrupt__"))


def _confirm(console, result, capture) -> bool:
    """Show the manifest and take a typed answer.

    The typed token rather than a bare y/n: `DELETE 7` cannot be produced by
    someone who has not read how many reports they are about to lose.
    """
    interrupt = result["__interrupt__"][0]
    requests = interrupt.value.get("action_requests", []) if isinstance(interrupt.value, dict) else []
    description = requests[0].get("description", "") if requests else ""
    render_confirmation(console, description)

    typed = console.input("[bold yellow]›[/bold yellow] ").strip()
    expected = capture.pending.token if capture.pending else "y"
    if typed == expected:
        return True

    console.print("[dim]Cancelled — nothing was deleted.[/dim]")
    return False


def _trace(console, deps, user, last_trace, command) -> None:
    """`/trace` explains the last turn; `/trace <id>` reads one back from
    storage, which is what makes a user's complaint a single lookup."""
    _, _, wanted = command.partition(" ")
    wanted = wanted.strip()
    if not wanted:
        render_trace(console, last_trace)
        return

    stored = deps.traces.get(owner_id=user, turn_id=wanted)
    if stored is None:
        console.print(f"No trace for turn {wanted}.")
        return
    render_trace(console, stored)
