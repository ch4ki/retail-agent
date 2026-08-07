"""Records the finished turn, on the way out.

The counterpart to `start_turn`. That node owns the head of a turn — blanking
scratch state so a failed turn cannot answer with the previous one's numbers —
and this owns the tail.

It exists because recording used to live in `cli/chat.py`, which made "is this
turn remembered?" depend on who invoked the graph rather than on what the graph
did. Studio invokes the compiled graph object directly and the eval harness
calls `run_turn`; neither passed through that line, so neither left a trace.

Sitting on every path out works because the graph already guarantees that every
path ends. One exception is worth naming: a turn paused at a breakpoint and then
abandoned never reaches the end, so it is never recorded. That was true before
this node existed too.
"""

from __future__ import annotations

import logging

from retail_agent.agent.deps import AgentDeps
from retail_agent.agent.state import TurnState
from retail_agent.obs.traces import from_state

log = logging.getLogger(__name__)


def finish_turn_node(state: TurnState, deps: AgentDeps) -> dict:
    """Persist the trace this turn accumulated.

    Returns no state update: the events are already in state, put there by the
    `_traced` wrapper as each node ran. This only moves them into storage.

    A stored trace therefore does not contain an event for this node. `_traced`
    appends after the wrapped function returns, so the write happens first. That
    is the right way round — a trace should show the seven nodes that did the
    work, not the bookkeeping that filed it — but it looks like an off-by-one
    to anyone comparing the stored trace against the returned state, which does
    carry it.
    """
    if not state.get("turn_id"):
        # Studio can submit state directly with no turn id, and a trace with no
        # id cannot be looked up later, so writing one would be noise.
        return {}

    try:
        deps.traces.record(from_state(state))
    except Exception as err:
        # A trace is a debugging aid. Failing to write one must never cost the
        # user their answer — the same rule the CLI applied before this moved.
        log.debug("trace not recorded: %s", err)

    return {}
