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
    render_definition_prompt,
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
from retail_agent.llm.provider import MissingCredentialsError, build_models
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

    # Must happen before the models are built: the tracer reads os.environ when
    # a model is constructed, not when it is called.
    tracing = configure_tracing(settings)

    try:
        llm, llm_fallbacks = build_models(settings)
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
        deps = build_deps(
            settings,
            llm=llm,
            llm_fallbacks=llm_fallbacks,
            source=source,
            console=console,
        )
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

        # Unconditional: every path out of `_answer` now produces a trace, and
        # falling back to the previous one is what made `/trace` describe a
        # question the user had not asked.
        last_trace = _answer(console, deps, saver, user, session_id, question)


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
    """Run one turn. Always returns its trace, including when it fails.

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
        # Armed here and nowhere else: a pause needs somebody who can answer it,
        # and this is the only caller with a person at a keyboard.
        agent = build_agent(
            deps, capture, checkpointer=saver, pause_for_definitions=True
        )
        with console.status("thinking…"):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}, config
            )

        # Two things pause the agent before a tool runs: a destructive call, and
        # a question whose terms nobody has settled. Both were resolved
        # read-only by the gate, so what is shown is exactly what stopped it.
        while _pending(result):
            decision = _decide(console, deps, result, capture, user)
            with console.status("working…"):
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
        # The recorder middleware is an `after_agent` hook and never runs when
        # the agent raises, so this is the only place a failed turn can be
        # written down. Without it the id above resolved to nothing and `/trace`
        # went on describing whichever turn last succeeded — the wrong question,
        # with no sign it was the wrong question.
        return _record_failure(deps, capture)

    answer = final_text(result)
    render_answer(console, answer, capture, prefs=preferred(deps.preferences, user))
    for field, value in capture.preference_changes:
        # Said by the CLI rather than left to the model: a setting that changed
        # without the reader being told is the failure this design cares about.
        console.print(f"[dim]Saved {field} = {value} as your default. /prefs to change it.[/dim]")
    # The trace was written by the recorder middleware, on every path out. This
    # is the same record, kept so `/trace` needs no round trip to storage.
    return capture.to_trace(answer)


def _record_failure(deps, capture):
    """The trace for a turn that died. Never raises.

    `failed` rather than left at `ok`: `self_correction_rate` counts a repaired
    turn as recovered only when it ended `ok`, so a turn that fixed its SQL and
    then died must not count as a self-correction that worked. That ratio is the
    only thing reading `status` now, and it is why the field survived `degraded`.
    """
    capture.status = "failed"
    trace = capture.to_trace("")
    try:
        deps.traces.record(trace)
    except Exception as err:
        # A failure while recording a failure must not become the failure.
        logging.getLogger(__name__).warning("could not record the failed turn (%s)", err)
    return trace


def _pending(result) -> bool:
    return bool(result.get("__interrupt__"))


def _decide(console, deps, result, capture, user) -> dict:
    """Turn a pause into the decision the middleware resumes with.

    Dispatching on the tool rather than on a mode flag: the two gates are
    configured in one middleware and either can be the reason a turn stopped, so
    the request itself is the only honest source of which one it was.
    """
    request = _first_request(result)
    if request.get("name") == "ask_for_definitions":
        return _settle_definitions(console, deps, capture, user, request)

    if _confirm(console, request.get("description", ""), capture):
        return {"type": "approve"}
    return {
        "type": "reject",
        "message": "The executive did not confirm. Nothing was deleted.",
    }


def _first_request(result) -> dict:
    interrupt = result["__interrupt__"][0]
    value = interrupt.value if isinstance(interrupt.value, dict) else {}
    requests = value.get("action_requests", [])
    return requests[0] if requests else {}


def _confirm(console, description, capture) -> bool:
    """Show the manifest and take a typed answer.

    The typed token rather than a bare y/n: `DELETE 7` cannot be produced by
    someone who has not read how many reports they are about to lose.
    """
    render_confirmation(console, description)

    typed = console.input("[bold yellow]›[/bold yellow] ").strip()
    expected = capture.pending.token if capture.pending else "y"
    if typed == expected:
        return True

    console.print("[dim]Cancelled — nothing was deleted.[/dim]")
    return False


# Returned by `_ask_definition` when the executive hands the decision back.
# A sentinel rather than a magic string, because any string they can type is a
# definition somebody might mean.
_HAND_BACK = object()


def _settle_definitions(console, deps, capture, user, request) -> dict:
    """Ask about each open term, then resume once.

    In question order and one at a time: each prompt stays a simple choice, and
    the options for the second term are generated knowing how the first was
    settled — otherwise a definition of `top` can quietly contradict the `loyal`
    just agreed.
    """
    from retail_agent.agent.schema import render_schema_outline
    from retail_agent.knowledge.proposals import propose

    pending = capture.pending_definition
    settled: dict[str, str] = {}
    # The outline, not the SQL rendering: the options are plain English, so the
    # values a column holds buy nothing and would cost a metadata scan per table
    # every time a term came up.
    schema = render_schema_outline(deps)

    for term in pending.terms if pending else ():
        options = propose(
            deps.llm,
            question=capture.question,
            term=term,
            schema=schema,
            settled=settled,
        )
        chosen = _ask_definition(console, term, options)

        if chosen is _HAND_BACK:
            # Applies to everything still open, not just this term: "you decide"
            # is not an answer that gets asked again for the next word.
            #
            # A reject rather than an edit. The tool body never runs, so this is
            # the only place that knows the terms went unanswered — hence the
            # recording here, which `assumption_note` reads on the way out to
            # force the disclosure into the answer.
            remaining = [t for t in pending.terms if t not in settled]
            capture.record_assumptions(remaining)
            return {
                "type": "reject",
                "message": (
                    "The executive asked you to decide. Do not ask again. "
                    f"Choose one concrete, defensible rule for: {', '.join(remaining)} "
                    "— a threshold, a window or a ranking — use it, and state "
                    "the rule you applied in your answer."
                ),
            }
        if not chosen:
            return {
                "type": "reject",
                "message": (
                    f"The executive did not define {term!r}, so nothing was "
                    "queried. Ask them what it should mean."
                ),
            }

        _remember(console, deps, user, term, chosen)
        settled[term] = chosen

    return {"type": "approve"}


def _ask_definition(console, term, options):
    """One term's answer: a definition, `_HAND_BACK`, or "" to cancel.

    A number outside the list is re-asked rather than taken literally — someone
    who types `9` at five options meant to pick something, and recording
    "loyal = 9" would be a definition they never gave.
    """
    from retail_agent.store.definitions import MAX_DEFINITION_CHARS

    own, hand_back = len(options) + 1, len(options) + 2
    while True:
        render_definition_prompt(console, term, options)
        typed = console.input("\n[bold yellow]›[/bold yellow] ").strip()

        if not typed:
            return ""
        if not typed.isdigit():
            return typed[:MAX_DEFINITION_CHARS]

        picked = int(typed)
        if 1 <= picked <= len(options):
            return options[picked - 1]
        if picked == own:
            console.print(f"[dim]What should {term!r} mean?[/dim]")
            return console.input("\n[bold yellow]›[/bold yellow] ").strip()[
                :MAX_DEFINITION_CHARS
            ]
        if picked == hand_back:
            return _HAND_BACK

        console.print(f"[yellow]Pick a number between 1 and {hand_back}.[/yellow]")


def _remember(console, deps, user, term, definition) -> None:
    """Keep the answer, and say so. Never fails the turn it just unblocked.

    Announced rather than left implicit: the executive has just changed what the
    agent will do on every future question containing this word, and `/definitions
    forget` is only a remedy if they know there is something to forget.
    """
    try:
        deps.definitions.remember(user_id=user, term=term, definition=definition)
    except Exception as err:
        logging.getLogger(__name__).warning(
            "could not save the definition of %r (%s)", term, err
        )
        console.print(
            f"[dim]Using {definition!r} for this question; I could not save it.[/dim]"
        )
        return

    console.print(f"[dim]Remembered: {term} = {definition}[/dim]")
    console.print(f"[dim]/definitions forget {term} to be asked again.[/dim]")


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
