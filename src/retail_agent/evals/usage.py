"""What a case cost, measured the same way on both arms.

A callback handler rather than instrumentation inside either agent. Both drive a
langchain `BaseChatModel`, so one handler attached through `config["callbacks"]`
sees every model call on the graph arm and on the ReAct arm alike. Counting them
separately — walking `events` on one side and the message list on the other —
would leave two implementations to disagree, and the disagreement would read as
a finding about the agents.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageCollector(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0

    def reset(self) -> None:
        """Called between cases. The suite reuses one handler across a run, and
        carrying case 3's tokens into case 4 would make later cases look
        progressively more expensive for no reason."""
        self.tokens_in = 0
        self.tokens_out = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for batch in response.generations:
            for generation in batch:
                usage = _usage(generation)
                # A provider that reports nothing costs the report a number.
                # Raising would cost it the answer too, and score a correct
                # case as an agent failure.
                self.tokens_in += int(usage.get("input_tokens") or 0)
                self.tokens_out += int(usage.get("output_tokens") or 0)


def _usage(generation: Any) -> dict:
    message = getattr(generation, "message", None)
    return getattr(message, "usage_metadata", None) or {}
