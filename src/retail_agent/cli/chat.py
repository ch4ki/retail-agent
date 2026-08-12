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
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import psycopg
import psycopg_pool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape

from retail_agent.agent.deps import TurnContext
from retail_agent.agent.supervisor import build_agent
from retail_agent.bootstrap import build_deps
from retail_agent.cli.render import (
    render_banner,
    render_confirmation,
    render_definition_prompt,
    render_definitions,
    render_error,
    render_footnote,
    render_metrics,
    render_persona,
    render_personas,
    render_preferences,
    render_reports,
    render_trace,
    render_trios,
)
from retail_agent.config import get_settings
from retail_agent.datasources.bigquery import BigQuerySource
from retail_agent.knowledge.trios import live_trios
from retail_agent.llm.errors import describe_llm_error
from retail_agent.llm.messages import message_text
from retail_agent.llm.provider import MissingCredentialsError, build_models
from retail_agent.obs.tracing import configure_tracing
from retail_agent.obs.traces import trace_from_state
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
  /prefs   how answers are rendered
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


# Losing the database should cost history, not the session. Losing a *name* —
# a NameError in startup wiring — must not look the same, which is what
# `except Exception` made it look like.
CONNECTION_ERRORS = (psycopg.OperationalError, psycopg_pool.PoolTimeout)


class _SchemaOutOfDate(Exception):
    """The checkpoint tables exist but predate the installed package's migrations.

    Raised by `_assert_migrated`, not by psycopg — there is no database error
    for this, since every column and table `SELECT_SQL` reads still exists.
    """


def _assert_migrated(saver) -> None:
    """Read one checkpoint that cannot exist, then confirm the schema version.

    `from_conn_string` opens a connection; it never touches the tables. So a
    database that was never migrated looks healthy here and fails on the first
    turn instead — inside `_answer`, where it surfaces as a failed turn rather
    than as the startup warning it is.

    The read alone is not enough to catch every un-migrated database, though:
    `get_tuple` only ever issues `SELECT_SQL`, and `SELECT_SQL` does not read
    every column a migration has added. `task_path` (added at migration index
    9, in the installed `langgraph-checkpoint-postgres`) is written by
    `INSERT_CHECKPOINT_WRITES_SQL`/`UPSERT_CHECKPOINT_WRITES_SQL` but never
    referenced by `SELECT_SQL`. A database sitting at an older migration
    version — created by an older version of this same dependency — would
    pass the read probe here and only fail once the agent tries to write,
    inside `_answer`, exactly the failure mode this function exists to avoid.
    So this also compares the version recorded in `checkpoint_migrations`
    against the version the installed `PostgresSaver.MIGRATIONS` expects.
    """
    saver.get_tuple({"configurable": {"thread_id": "__migration_probe__"}})

    expected = len(saver.MIGRATIONS) - 1
    with saver._cursor() as cur:
        cur.execute("SELECT max(v) AS v FROM checkpoint_migrations")
        row = cur.fetchone()
    applied = row["v"] if row and row["v"] is not None else -1
    if applied < expected:
        raise _SchemaOutOfDate(applied, expected)


@contextmanager
def _checkpointer(console: Console, database_url: str):
    """Durable conversation state, degrading to in-memory if Postgres is down.

    A missing database should cost you history across restarts, not the ability
    to use the agent. Anything outside the two named states is a bug and
    propagates.
    """
    with ExitStack() as stack:
        try:
            saver = stack.enter_context(PostgresSaver.from_conn_string(database_url))
            _assert_migrated(saver)
        except (
            psycopg.errors.UndefinedTable,
            psycopg.errors.UndefinedColumn,
            psycopg.errors.InsufficientPrivilege,
            _SchemaOutOfDate,
        ) as err:
            # These three are the genuine "the schema is missing or wrong"
            # states. Deliberately narrower than the base `psycopg.
            # ProgrammingError`: a malformed `DATABASE_URL` (e.g. a stray
            # query parameter) also raises a bare `ProgrammingError` at
            # conninfo-parse time, before any connection is attempted — that
            # is a config bug, not a schema problem, and "run `retail-agent
            # migrate`" would not fix it, so it must keep propagating. It is
            # also not a broad `Exception` catch: a NameError from broken
            # startup wiring is not a ProgrammingError and still propagates.
            extra = ""
            if isinstance(err, _SchemaOutOfDate):
                extra = f" (applied={err.args[0]}, expected={err.args[1]})"
            logging.getLogger(__name__).debug(
                "checkpoint schema missing or out of date: %s%s", err, extra
            )
            stack.close()
            console.print(
                "[yellow]The database's checkpoint schema is missing or out "
                "of date — this session will not be saved. Run "
                "`retail-agent migrate` to enable history.[/yellow]"
            )
        except CONNECTION_ERRORS as err:
            # Full detail only under --verbose; the console gets one actionable line.
            logging.getLogger(__name__).debug("postgres unavailable: %s", err)
            console.print(
                "[yellow]Postgres unreachable — this session will not be saved. "
                "Run `docker compose up -d postgres` to enable history.[/yellow]"
            )
        else:
            # Outside the `try`, so an exception from the REPL body propagates
            # instead of being thrown back in here and read as a database fault.
            yield saver
            return

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

    from retail_agent.store.preferences import notes_for

    parts = command.split()
    if len(parts) == 1:
        render_preferences(
            console,
            preferred(deps.preferences, user),
            notes_for(deps.preferences, user),
        )
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
    render_preferences(console, updated, notes_for(deps.preferences, user))


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


# The supervisor's own graph node, as `create_agent` names it — confirmed
# against a live stream, not assumed: `report_writer` and `ask_about_report`
# both make a bare `model.invoke()` inside `resilient_call`, and LangGraph
# propagates the parent run's callback context into any Runnable invoked
# synchronously inside a node, so those internal calls surface as `AIMessage`
# chunks on `stream_mode="messages"` too — tagged `langgraph_node="tools"`,
# never touching `TurnState["messages"]`. A probe against the real agent
# (report_writer scripted, then the covering sentence) showed exactly this:
#   AIMessage  node=model  text=''                          <- tool-call request
#   AIMessage  node=tools  text='# Q1 Report\n\nRevenue...'  <- the leak
#   ToolMessage node=tools text="Report ... written ..."     <- already skipped by type
#   AIMessage  node=model  text="Here's your report."        <- the actual answer
# The `analyst` subagent does not leak this way: it is a compiled subgraph
# reached through `.invoke()`, which LangGraph does not auto-forward into the
# parent's message stream.
_SUPERVISOR_NODE = "model"


def _stream_turn(console, agent, payload, config, context) -> str:
    """Drive one turn, rendering as it arrives. Returns the answer text.

    `messages` carries the supervisor's tokens; `custom` carries whatever a tool
    chose to say while it worked. Only `AIMessage` chunks with content, from the
    supervisor's own node, are the answer:

    - A `ToolMessage` is the tool's receipt to the model, and `report_writer`'s
      receipt is deliberately a covering sentence rather than the report it
      wrote — skipped by type, not by guessing from content.
    - An `AIMessage` from any node other than `_SUPERVISOR_NODE` is a tool's
      own internal model call leaking through the shared callback context (see
      `_SUPERVISOR_NODE`'s comment) — skipped by node, since `isinstance`
      alone cannot tell it apart from the real answer.

    Two more things a per-chunk stream gets wrong if handled the same way a
    whole message is:

    - `message_text` ends in `.strip()`, correct for a whole reply but wrong
      per chunk — stripping every chunk eats the space it carries at its own
      edge, and concatenating stripped chunks glues words together with none
      between them. `chunk_text` is the same content-block handling without
      the strip.
    - The supervisor's own node can run more than once in a turn — a real
      model routinely narrates before a tool call ("Let me look that up.")
      and again before the next, and each of those is its own `AIMessage`,
      not a continuation of the last. `parts` resets whenever `message.id`
      changes, so the answer this returns is the FINAL supervisor message —
      the one `final_text(agent.get_state(config).values)` would also read —
      not every one of them glued together. The narration still reaches the
      console as it streams; it just does not end up in the returned answer.
    """
    from langchain_core.messages import AIMessage

    from retail_agent.llm.messages import chunk_text

    parts: list[str] = []
    current_message_id: object = None
    mid_line = False
    # Every `AIMessage` chunk's node, seen regardless of the filter below —
    # if the supervisor's own node is never among them, the name this filter
    # relies on has changed, and silently rendering nothing would be a worse
    # bug than the one this replaced.
    seen_nodes: set[str] = set()

    # Held until the first thing reaches the screen, then dropped. Removing
    # the whole-turn spinner was right — it hid three queries and a report
    # behind one unchanging line — but it also removed the only feedback
    # before the first token, and a turn that opens with an `analyst` call
    # says nothing until that subagent does. `ExitStack.close()` is
    # idempotent, so each render site below can drop it without knowing
    # whether it is the first.
    waiting = ExitStack()
    waiting.enter_context(console.status("thinking…"))
    try:
        for mode, chunk in agent.stream(
            payload, config, context=context, stream_mode=["messages", "custom"]
        ):
            if mode == "custom":
                # First output of the turn? The wait is over.
                waiting.close()
                if mid_line:
                    # The last thing printed was an answer or narration token,
                    # left mid-line by `end=""` below. Left alone, this progress
                    # line would glue onto it: "Let me look that up.1 row(s)...".
                    console.print()
                    mid_line = False
                # `markup=False, highlight=False`: `chunk["progress"]` can embed
                # a tool's own text (a report title, a search term) verbatim, and
                # that text is not console markup. Unescaped, a stray `[/bold]`
                # in it does not just print wrong — it raises `MarkupError` and
                # takes the whole turn down with it (see the answer print below).
                # `style="dim"` gets the same look the old `[dim]...[/dim]`
                # wrapping gave, without parsing the interpolated text as markup.
                console.print(
                    str(chunk.get("progress", chunk)),
                    style="dim",
                    markup=False,
                    highlight=False,
                )
            elif mode == "messages":
                message, meta = chunk
                if isinstance(message, AIMessage):
                    # `str(...)`: a chunk with no `langgraph_node` at all must still
                    # land somewhere sortable in `seen_nodes` below, not raise a
                    # `TypeError` from comparing `None` against a string.
                    node = str(meta.get("langgraph_node"))
                    seen_nodes.add(node)
                    if node == _SUPERVISOR_NODE:
                        if message.id != current_message_id:
                            # A new supervisor message has started — narration
                            # from an earlier one in this same turn is not part
                            # of the answer this function returns.
                            #
                            # Depends on every chunk of one model call sharing a
                            # stable id and a new call getting a different one —
                            # true for all four configured providers, and true in
                            # general because `langchain_core`'s own streaming
                            # loop stamps an id-less chunk with the call's
                            # `run_id` (`chat_models.py`, `_generate_with_cache`:
                            # `if chunk.message.id is None: chunk.message.id =
                            # run_id`). It stops being true for a provider whose
                            # *own* chunks already carry ids that change mid-call
                            # — OpenAI's Responses API does this (e.g.
                            # `resp_1` on early chunks, then `lc_run--x` on
                            # later ones of the SAME reply) — because
                            # `langchain_core` only fills in a missing id, it
                            # never overrides one a provider already set. On such
                            # a provider this silently drops every chunk before
                            # the id change instead of resetting on a real
                            # message boundary. Not reachable with the providers
                            # `llm/provider.py` builds today; re-check this
                            # assumption before pointing `LLM_PROVIDER`/
                            # `LLM_MODEL` at a Responses-API-shaped model.
                            parts = []
                            current_message_id = message.id
                        text = chunk_text(message)
                        if text:
                            parts.append(text)
                            # `markup=False, highlight=False`: the model's own
                            # words are not console markup either, for the same
                            # reason as the progress line above — see C3 in the
                            # streaming-the-turn review.
                            # First output of the turn? The wait is over.
                            waiting.close()
                            console.print(text, end="", markup=False, highlight=False)
                            mid_line = True
    finally:
        waiting.close()

    if seen_nodes and _SUPERVISOR_NODE not in seen_nodes:
        raise RuntimeError(
            f"expected an AIMessage chunk from node {_SUPERVISOR_NODE!r}, saw "
            f"only {sorted(seen_nodes)} — the supervisor's node name changed "
            "and _stream_turn's filter needs updating, not silent dropping."
        )
    if mid_line:
        console.print()
    return "".join(parts)


def _answer(console, deps, saver, user, session_id, question):
    """Run one turn. Always returns its trace, including when it fails.

    A fresh agent and a fresh `TurnContext` with a freshly minted `turn_id`
    per turn. The persona and preferences are read per model call, so
    rebuilding costs nothing and rules out a stale binding.
    """
    config = {"configurable": {"thread_id": session_id}}
    turn_id = uuid.uuid4().hex[:12]
    context = TurnContext(user_id=user, session_id=session_id, turn_id=turn_id)
    # Set only once the agent is built: `_record_failure` reads the
    # checkpoint back through it, and a turn that died before it could be
    # built has no checkpoint to read either way.
    agent = None

    try:
        # Inside the guard, not above it: assembling the agent reads the persona
        # store and the tool list, and a REPL that dies before its own error
        # handler is a REPL that dies.
        # Armed here and nowhere else: a pause needs somebody who can answer it,
        # and this is the only caller with a person at a keyboard.
        agent = build_agent(deps, checkpointer=saver, pause_for_definitions=True)
        answer = _stream_turn(
            console,
            agent,
            {"messages": [{"role": "user", "content": question}]},
            config,
            context,
        )
        result = agent.get_state(config).values

        # Two things pause a turn: `delete_reports` and `ask_for_definitions`,
        # both interrupting themselves now, so `_decide` returns exactly the
        # value each one expects back — nothing here has to reshape it.
        while _pending(agent, config):
            resume = _decide(console, deps, agent, config)
            answer = _stream_turn(
                console, agent, Command(resume=resume), config, context
            )
            result = agent.get_state(config).values
    except Exception as err:  # the REPL must survive anything
        # Full detail goes to the log; the user gets one actionable line. The
        # turn id makes a complaint a single lookup rather than an
        # investigation, so it goes on screen and not only into the log.
        logging.getLogger(__name__).exception("turn failed")
        render_error(
            console,
            describe_llm_error(err, provider=deps.settings.llm_provider),
            turn_id=context.turn_id,
        )
        # The recorder middleware is an `after_agent` hook and never runs when
        # the agent raises, so this is the only place a failed turn can be
        # written down. Without it the id above resolved to nothing and `/trace`
        # went on describing whichever turn last succeeded — the wrong question,
        # with no sign it was the wrong question.
        return _record_failure(deps, agent, config, context)

    # `_stream_turn` already printed `answer` live, token by token, as it
    # arrived — calling `render_answer` here would print it a second time
    # through `Markdown`. Only the footnote (masking, repaired queries) still
    # needs rendering, and only when there was an answer to attach it to,
    # matching `render_answer`'s own `if not answer: return`.
    if answer:
        render_footnote(console, result, prefs=preferred(deps.preferences, user))
    # Printed here rather than left to the model, for the same reason the
    # preference note below is: what the executive was told is not something to
    # leave to whether the model remembered to say it. Below the answer, which
    # is the covering sentence introducing it.
    render_reports(console, _written_reports(deps, result, user))
    for change in result.get("preference_changes") or []:
        # Said by the CLI rather than left to the model: a setting that changed
        # without the reader being told is the failure this design cares about.
        # `change["note"]` is `note_preference`'s own text — the same
        # untrusted, model-paraphrased note `render_preferences` shows and
        # escapes later, via `/prefs` — so it is escaped here too, same
        # reason as every other model/store text this file no longer trusts
        # as console markup (see `_stream_turn` and `render_error`).
        # `action` is one of two literals this codebase writes ("added",
        # "removed"), not model text.
        console.print(
            f"[dim]Saved {change['action']} = {escape(change['note'])} as your "
            f"default. /prefs to change it.[/dim]"
        )
    # The trace was written by the recorder middleware, on every path out. This
    # is the same record, read back off the same state rather than recomputed.
    return trace_from_state(
        result,
        answer,
        user_id=context.user_id,
        session_id=context.session_id,
        turn_id=context.turn_id,
    )


def _written_reports(deps, state, user):
    """This turn's reports, with the body read back from the store.

    `TurnState["reports_written"]` never carries a body — state is
    checkpointed on every super-step, and the library's copy is the one that
    gets read (see `reports.py`) — so the CLI has to fetch it before it can
    print the report it just wrote.
    """
    written = []
    for entry in state.get("reports_written") or []:
        report = deps.reports.get(owner_id=user, report_id=entry["report_id"])
        if report is None:
            continue
        written.append({**entry, "body": report.body})
    return written


def _record_failure(deps, agent, config, context):
    """The trace for a turn that died. Never raises.

    Recovers the turn's record from the checkpointer rather than from a
    return value the graph never produced — `agent.invoke` raised, so there is
    no result to read `attempts`/`events`/... off directly. Reading the
    checkpoint instead means a turn that ran two queries and then died still
    records those two attempts, and it survives the process dying with it,
    which holding the record in memory never did.

    `agent` is `None` when the turn died before it could even be built (a
    startup wiring fault, say) — there is no checkpoint to read then, and the
    trace is built from an empty state, same as a turn that never ran a tool.

    `failed` rather than left at `ok`: `self_correction_rate` counts a repaired
    turn as recovered only when it ended `ok`, so a turn that fixed its SQL and
    then died must not count as a self-correction that worked. That ratio is
    the only thing reading `status` now, and it is why the field survived
    `degraded`.

    Takes the `TurnContext` the turn was invoked with: identity lives there,
    not on anything the graph accumulates, and a turn that died before the
    recorder ran still needs its `user_id`/`session_id`/`turn_id` for the trace.
    """
    state = {}
    if agent is not None:
        try:
            state = agent.get_state(config).values or {}
        except Exception as err:
            logging.getLogger(__name__).warning(
                "could not recover the checkpoint for the failed turn (%s)", err
            )

    trace = replace(
        trace_from_state(
            state,
            "",
            user_id=context.user_id,
            session_id=context.session_id,
            turn_id=context.turn_id,
        ),
        status="failed",
    )
    try:
        deps.traces.record(trace)
    except Exception as err:
        # A failure while recording a failure must not become the failure.
        logging.getLogger(__name__).warning("could not record the failed turn (%s)", err)
    return trace


def _pending(agent, config) -> bool:
    """Whether the turn stopped for a person.

    `agent.get_state(config).next` rather than an `__interrupt__` key: a
    streamed turn's result is the state read back from the checkpointer, and
    this is the same check `seams.ask_once` already uses.
    """
    return bool(agent.get_state(config).next)


def _payload(snapshot) -> dict:
    """What the tool asked for.

    Reads `snapshot.interrupts` rather than a `result["__interrupt__"]` key:
    `agent.invoke` used to stamp that key onto its return value, but a streamed
    turn's `result` is the checkpointed state, which never carries it — the
    interrupt itself only lives on the state snapshot `get_state` returns.
    """
    # Not a state key: `__interrupt__` is only ever stamped onto `invoke()`'s
    # return value, never written into the checkpoint, so there is no
    # `snapshot.values["__interrupt__"]` to fall back to here.
    if not snapshot.interrupts:
        return {}
    value = snapshot.interrupts[0].value
    return value if isinstance(value, dict) else {}


def _question_from(result) -> str:
    """This turn's own question, the last `HumanMessage` in the thread.

    Identity and the transcript both live in graph state now, so this reads
    the same fact `memory.py`'s own `_last_human_text` and `obs/traces.py`'s
    `_last_human_text` do, rather than a value synced onto a closure at the
    top of every turn.
    """
    from langchain_core.messages import HumanMessage

    for message in reversed(result.get("messages") or []):
        if isinstance(message, HumanMessage):
            return message_text(message)
    return ""


def _decide(console, deps, agent, config) -> dict:
    """Turn a pause into the value the tool resumes with.

    Dispatching on the payload's own `kind`: the tool that asked is the tool
    that named the question, so there is nothing to infer.

    Takes `agent`/`config` rather than a result dict: the interrupt payload
    lives on the state snapshot (see `_payload`), not on the checkpointed
    values a streamed turn's `result` now is.
    """
    snapshot = agent.get_state(config)
    payload = _payload(snapshot)
    if payload.get("kind") == "ask_for_definitions":
        return _settle_definitions(
            console, deps, _question_from(snapshot.values), payload
        )

    if _confirm(console, payload):
        return {
            "approved": True,
            "report_ids": list(payload["report_ids"]),
            "token": payload["token"],
        }
    return {"approved": False}


def _confirm(console, payload) -> bool:
    """Show the manifest and take a typed answer.

    The typed token rather than a bare y/n: `DELETE 7` cannot be produced by
    someone who has not read how many reports they are about to lose.
    """
    render_confirmation(console, payload.get("manifest", ""))

    typed = console.input("[bold yellow]›[/bold yellow] ").strip()
    if typed == payload.get("token", "y"):
        return True

    console.print("[dim]Cancelled — nothing was deleted.[/dim]")
    return False


# Returned by `_ask_definition` when the executive hands the decision back.
# A sentinel rather than a magic string, because any string they can type is a
# definition somebody might mean.
_HAND_BACK = object()


def _settle_definitions(console, deps, question, payload) -> dict:
    """Ask about each open term, then resume once.

    In question order and one at a time: each prompt stays a simple choice, and
    the options for the second term are generated knowing how the first was
    settled — otherwise a definition of `top` can quietly contradict the `loyal`
    just agreed.

    Collects answers rather than writing them: the CLI must not write to the
    store before resuming, because the tool body replays on resume and its own
    lookup would then find a store that already changed, leaving the
    `interrupt()` call unreachable and this resume value unconsumed. So the
    answers travel back in the resume value, and `ask_for_definitions` is what
    stores them.

    `question` is this turn's own question, read off the graph state by the
    caller (`_decide`) — identity and the transcript both live there now,
    not on anything built and held across the turn.
    """
    from retail_agent.agent.schema import render_schema_outline
    from retail_agent.knowledge.proposals import propose

    terms = payload.get("terms") or []
    answers: dict[str, str] = {}
    # The outline, not the SQL rendering: the options are plain English, so the
    # values a column holds buy nothing and would cost a metadata scan per table
    # every time a term came up.
    schema = render_schema_outline(deps)

    for term in terms:
        options = propose(
            deps,
            question=question,
            term=term,
            schema=schema,
            settled=answers,
        )
        chosen = _ask_definition(console, term, options)

        if chosen is _HAND_BACK:
            # Applies to everything still open, not just this term: "you decide"
            # is not an answer that gets asked again for the next word. The
            # remaining terms — this one and any after it — are simply absent
            # from `answers`; the tool records them as assumed on resume.
            return {"answers": answers}
        if not chosen:
            # Stops asking, but does not undo what was already agreed: a term
            # settled earlier in this same batch was answered, and cancelling
            # on a later one must not erase it — only what's still open here
            # goes unanswered, same as a hand-back.
            return {"answers": answers}

        answers[term] = chosen

    return {"answers": answers}


def _ask_definition(console, term, options):
    """One term's answer: a definition, `_HAND_BACK`, or "" to cancel.

    A number outside the list is re-asked rather than taken literally — someone
    who types `9` at five options meant to pick something, and recording
    "loyal = 9" would be a definition they never gave.

    A slash command is re-asked for the same reason, and it is the one that
    actually happened: `/persona` typed here was stored as the meaning of
    "loyal customers", and every later turn read it back, reported the term
    settled, and handed the model a definition that says nothing. That is worse
    than never having asked, because the only remedy is a `/definitions forget`
    the executive has no reason to know they need.
    """
    own, hand_back = len(options) + 1, len(options) + 2
    while True:
        render_definition_prompt(console, term, options)
        typed = console.input("\n[bold yellow]›[/bold yellow] ").strip()

        if not typed:
            return ""
        if not typed.isdigit():
            accepted = _vet_definition(console, term, typed)
            if accepted is None:
                continue
            return accepted

        picked = int(typed)
        if 1 <= picked <= len(options):
            return options[picked - 1]
        if picked == own:
            console.print(f"[dim]What should {term!r} mean?[/dim]")
            while True:
                typed = console.input("\n[bold yellow]›[/bold yellow] ").strip()
                if not typed:
                    return ""
                accepted = _vet_definition(console, term, typed)
                if accepted is not None:
                    return accepted
        if picked == hand_back:
            return _HAND_BACK

        console.print(f"[yellow]Pick a number between 1 and {hand_back}.[/yellow]")


def _vet_definition(console, term, typed) -> str | None:
    """Free text on its way to becoming a definition, or None to re-ask.

    Every path that accepts typed text goes through here — the slash-command
    incident recurred through the "write my own" prompt precisely because the
    guard sat on only one of the two input sites. The echo is escaped: `[/]`
    in a typed command is a Rich closing tag, and unescaped it kills the
    interrupt-resume flow with a MarkupError instead of re-asking.
    """
    from rich.markup import escape

    from retail_agent.store.definitions import MAX_DEFINITION_CHARS

    if typed.startswith("/"):
        console.print(
            f"[yellow]Commands do not run here. Answer for {escape(term)!r}, "
            f"or press enter to cancel and then type {escape(typed)}.[/yellow]"
        )
        return None
    return typed[:MAX_DEFINITION_CHARS]


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
