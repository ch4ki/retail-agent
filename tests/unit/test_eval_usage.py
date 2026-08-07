"""Counting what an answer cost.

One handler for both arms. Per-arm instrumentation is how a measurement ends up
flattering the thing it was written alongside, and cost is half the claim being
tested — an arm that wins on accuracy while spending three times the tokens has
not obviously won.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from retail_agent.evals.usage import UsageCollector


def reply(*, input_tokens: int | None, output_tokens: int | None) -> LLMResult:
    usage = None
    if input_tokens is not None:
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + (output_tokens or 0),
        }
    message = AIMessage(content="...", usage_metadata=usage)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_a_fresh_collector_has_spent_nothing():
    collector = UsageCollector()

    assert collector.tokens_in == 0
    assert collector.tokens_out == 0


def test_usage_accumulates_across_every_model_call_in_a_turn():
    """A turn is several calls on both arms — plan, draft, repair, synthesize on
    the graph; one per ReAct step. The case's cost is their sum."""
    collector = UsageCollector()

    collector.on_llm_end(reply(input_tokens=100, output_tokens=20))
    collector.on_llm_end(reply(input_tokens=50, output_tokens=10))

    assert collector.tokens_in == 150
    assert collector.tokens_out == 30


def test_a_provider_that_reports_no_usage_does_not_fail_the_case():
    """Not every provider returns usage metadata. Losing the token count is a
    gap in the report; raising here would lose the answer as well, and score a
    correct case as an agent error."""
    collector = UsageCollector()

    collector.on_llm_end(reply(input_tokens=None, output_tokens=None))

    assert collector.tokens_in == 0


def test_the_collector_resets_between_cases():
    """Cases run sequentially through one suite. A collector that carried case
    3's tokens into case 4 would make later cases look progressively worse."""
    collector = UsageCollector()
    collector.on_llm_end(reply(input_tokens=100, output_tokens=20))

    collector.reset()

    assert collector.tokens_in == 0
    assert collector.tokens_out == 0
