"""The supervisor's middleware stack, as a list whose order is a guarantee."""

from dataclasses import replace

from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately

from retail_agent.agent.capture import TurnCapture
from retail_agent.agent.middleware import _summarization, supervisor_middleware
from retail_agent.agent.prompts import CONVERSATION_SUMMARY_PROMPT


def tuned(deps, settings, **overrides):
    # `Settings` is a pydantic `BaseSettings`, not a `dataclasses.dataclass` —
    # `dataclasses.replace` raises `TypeError` on it under this Python version,
    # so it gets pydantic's own copy-with-overrides instead. `AgentDeps` is a
    # real dataclass, so `replace` still applies there.
    return replace(deps, settings=settings.model_copy(update=overrides))


def stack(deps):
    return supervisor_middleware(deps, TurnCapture(user_id="exec"))


def test_the_summariser_never_runs_before_pii_redaction(make_deps, settings):
    """Both hook `before_model`, and those run in list order. The summariser
    writes its output back to state, so going first would read unredacted
    warehouse output and persist it — a transient exposure made permanent."""
    deps = tuned(make_deps(), settings, context_summarize_trigger_tokens=1_000)

    kinds = [type(m) for m in stack(deps)]
    last_pii = max(i for i, kind in enumerate(kinds) if kind is PIIMiddleware)
    summariser = kinds.index(SummarizationMiddleware)

    assert last_pii < summariser


def test_zero_disables_the_summariser_entirely(make_deps, settings):
    """Off means absent, not present-and-inert: the stack is the only place
    this agent's control flow is visible, and something listed there that
    cannot act makes it a worse description of the turn."""
    deps = tuned(make_deps(), settings, context_summarize_trigger_tokens=0)

    assert not any(isinstance(m, SummarizationMiddleware) for m in stack(deps))


def test_a_thread_past_the_trigger_is_compacted_and_keeps_its_tail(
    make_deps, settings
):
    """The point of the mechanism: a session on a small-window model has to
    stay usable, and the most recent turns are the ones still being discussed."""
    deps = tuned(
        make_deps(script=["## What the executive asked about\nDenim performance."]),
        settings,
        context_summarize_trigger_tokens=200,
        context_keep_messages=4,
    )
    messages = []
    for turn in range(40):
        messages.append(HumanMessage(content=f"question {turn} about denim sales"))
        messages.append(AIMessage(content=f"answer {turn}, with some length to it"))

    result = _summarization(deps).before_model({"messages": messages}, None)

    # LangGraph's reducer protocol, not a plain replacement: the first entry is
    # a `RemoveMessage(id='__remove_all__')` telling the state reducer to clear
    # the existing thread before appending what follows. It carries no tokens
    # of its own, and `count_tokens_approximately` has no case for it.
    compacted = [m for m in result["messages"] if not isinstance(m, RemoveMessage)]
    assert count_tokens_approximately(compacted) < count_tokens_approximately(messages)
    assert any(m.content == messages[-1].content for m in compacted)


def test_the_summary_prompt_forbids_restating_figures():
    """A summariser that retypes a number produces, on the next turn, a figure
    the model believes came from a query. That is the failure SAFETY_RULES
    exists to prevent."""
    assert "ARTIFACTS" not in CONVERSATION_SUMMARY_PROMPT
    assert "NEXT STEPS" not in CONVERSATION_SUMMARY_PROMPT
    assert "do not restate any figure" in CONVERSATION_SUMMARY_PROMPT
    assert "{messages}" in CONVERSATION_SUMMARY_PROMPT


def test_the_recorder_measures_the_thread_it_just_finished(make_deps):
    """The threshold in Settings was set against a number nothing measured.
    This is that number."""
    deps = make_deps()
    capture = TurnCapture(user_id="exec")
    # `_resilience` appends after the recorder, and with no fallbacks
    # configured it still appends `ModelRetryMiddleware` — so the recorder is
    # second-to-last, not last.
    recorder = supervisor_middleware(deps, capture)[-2]
    state = {
        "messages": [
            HumanMessage(content="How did denim do in Q1?"),
            AIMessage(content="Denim revenue fell 11.8% in Q1."),
        ]
    }

    recorder.after_agent(state, None)

    assert capture.context_tokens > 0
    assert deps.traces.recent(owner_id="exec")[0].context_tokens == capture.context_tokens
