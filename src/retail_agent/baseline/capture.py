"""What the ReAct arm remembers of its own tool calls.

The graph keeps `frames` and `sql_attempts` in state, and `answer_from_state`
reads the number back out of them. A ReAct agent has no equivalent: it has a
message list, and the rows it saw were rendered into a `ToolMessage` string.

Re-parsing that string to score it would measure how the tool formatted its
output, which is the same mistake `runner.py` already documents about parsing
prose. So the tools write the real `MaskedFrame` here on the way past, and the
eval reads it back untouched.

One capture per turn, created by `seams.ask` and closed over by the tools. It is
deliberately not global: eval cases run sequentially today, but a shared capture
would silently attribute case 4's rows to case 3 the moment that changed.
"""

from __future__ import annotations

from collections.abc import Sequence

from retail_agent.agent.state import MaskedFrame


class ResultCapture:
    def __init__(self) -> None:
        self._sql = ""
        self._frame: MaskedFrame | None = None
        self._trio_ids: list[str] = []
        self.calls = 0

    def record_query(self, sql: str, frame: MaskedFrame) -> None:
        """Only successful executions reach here.

        A guard violation or a warehouse error raises inside the tool and is
        handed back to the model as an error `ToolMessage`, so a repaired turn
        leaves only the query that actually ran — which is the one being scored.
        """
        self._sql = sql
        self._frame = frame
        self.calls += 1

    def record_definitions(self, trio_ids: Sequence[str]) -> None:
        # Order-preserving union: the model may look the same term up twice,
        # and `AgentAnswer.trios` should report what was consulted, not how
        # often. Lists rather than a set so the report reads in call order.
        for trio_id in trio_ids:
            if trio_id not in self._trio_ids:
                self._trio_ids.append(trio_id)
        self.calls += 1

    @property
    def frame(self) -> MaskedFrame | None:
        return self._frame

    @property
    def executed_sql(self) -> str:
        return self._sql

    @property
    def trio_ids(self) -> tuple[str, ...]:
        return tuple(self._trio_ids)
