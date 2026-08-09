"""A scripted model and warehouse so agent behaviour can be tested offline."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from retail_agent.agent.deps import AgentDeps
from retail_agent.config import Settings
from retail_agent.datasources.base import (
    ColumnSchema,
    DryRunResult,
    QueryResult,
    QuerySyntaxError,
    TableSchema,
)
from retail_agent.safety.pii import PiiPolicy
from retail_agent.obs.traces import InMemoryTraceStore
from retail_agent.store.memory_reports import InMemoryReportStore


class ScriptExhausted(BaseException):
    """Deliberately not an `Exception`.

    Nodes catch `Exception` around model calls to degrade gracefully. If the
    fake raised one, running out of replies would look like a provider failure
    and the test would quietly pass through the fallback path instead of
    failing. This one propagates through those handlers.
    """


class ScriptedLLM:
    """Returns queued replies in order. Records every prompt it received.

    `blocks=True` reproduces the shape Gemini actually returns — a list of
    content blocks carrying thinking signatures — rather than a plain string.
    Providers disagree here, and a string-only double hides real bugs.
    """

    def __init__(self, replies: list[str], blocks: bool = False):
        self.replies = list(replies)
        self.blocks = blocks
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        reply = self._next(messages)
        if self.blocks:
            return AIMessage(
                content=[
                    {"type": "text", "text": reply, "extras": {"signature": "sig"}}
                ]
            )
        return AIMessage(content=reply)

    def with_structured_output(self, schema, **kwargs):
        """Mirror the real runnable: the caller gets a validated model back.

        Queue a dict (or an instance) for these calls. Validation is real, so a
        reply the schema rejects fails the test rather than reaching the node.
        """
        return _ScriptedStructured(self, schema)

    def _next(self, messages):
        self.prompts.append(_as_text(messages))
        if not self.replies:
            raise ScriptExhausted("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


class _ScriptedStructured:
    def __init__(self, llm: "ScriptedLLM", schema):
        self._llm = llm
        self._schema = schema

    def invoke(self, messages, **kwargs):
        reply = self._llm._next(messages)
        if isinstance(reply, self._schema):
            return reply
        if isinstance(reply, dict):
            return self._schema(**reply)
        return self._schema.model_validate_json(reply)


def _as_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(getattr(m, "content", m)) for m in messages)


class ScriptedChatModel(BaseChatModel):
    """A real `BaseChatModel` that replays queued turns through `create_agent`.

    `ScriptedLLM` cannot be used here: `create_agent` calls `bind_tools`, which
    `BaseChatModel` leaves unimplemented, and the agent loop reads
    `AIMessage.tool_calls` rather than content. So a script entry is either a
    string (a final answer) or a list of `(tool_name, args)` pairs (one round of
    tool calls).

    Every prompt is recorded, because several tests assert on what the model was
    *given* — that the persona reached it, that the safety rules did — and those
    are the assertions a mock returning canned text cannot make.
    """

    script: list = []
    prompts: list = []
    bound_tools: list = []

    def __init__(self, script: list, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Assigned after init so each instance owns its lists rather than
        # sharing the class attribute pydantic would otherwise hand out.
        object.__setattr__(self, "script", list(script))
        object.__setattr__(self, "prompts", [])
        object.__setattr__(self, "bound_tools", [])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        object.__setattr__(
            self, "bound_tools", [getattr(t, "name", getattr(t, "__name__", t)) for t in tools]
        )
        return self

    def with_structured_output(self, schema, **kwargs):
        """Queue a dict for these calls, as `ScriptedLLM` does.

        `BaseChatModel` would otherwise route this through `bind_tools` and
        parse a tool call back out, so a script entry meant for a structured
        call would have to be written as a tool-call pair. Validation stays
        real: a reply the schema rejects fails the test rather than reaching
        the caller.
        """
        return _ScriptedStructuredChat(self, schema)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append("\n".join(str(m.content) for m in messages))
        if not self.script:
            raise ScriptExhausted("ScriptedChatModel ran out of turns")

        turn = self.script.pop(0)
        if isinstance(turn, str):
            message = AIMessage(content=turn)
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call_{index}"}
                    for index, (name, args) in enumerate(turn)
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _ScriptedStructuredChat:
    def __init__(self, model: "ScriptedChatModel", schema):
        self._model = model
        self._schema = schema

    def invoke(self, messages, **kwargs):
        self._model.prompts.append(_as_text(messages))
        if not self._model.script:
            raise ScriptExhausted("ScriptedChatModel ran out of turns")
        reply = self._model.script.pop(0)
        if isinstance(reply, self._schema):
            return reply
        if isinstance(reply, dict):
            return self._schema(**reply)
        return self._schema.model_validate_json(reply)


@dataclass
class FakeSource:
    """Serves canned frames; raises for SQL containing a marker in `failing`."""

    frames: dict[str, pd.DataFrame]
    failing: set[str] = field(default_factory=set)
    # SQL containing one of these runs fine and returns nothing — the shape of
    # a literal that does not match, which is not an error anywhere.
    empty_for: set[str] = field(default_factory=set)
    # The true size of the result when the read was capped; None means the
    # frame is the whole result.
    total_rows: int | None = None
    # The same non-match through an aggregate: SUM() over no rows returns one
    # row holding NULL. That is what BigQuery actually returns, and it is the
    # shape the live failure took.
    null_aggregate_for: set[str] = field(default_factory=set)
    dialect: str = "bigquery"
    executed: list[str] = field(default_factory=list)

    def list_tables(self):
        return ["order_items", "orders", "products", "users"]

    def describe(self, table):
        return TableSchema(
            name=table,
            columns=(
                ColumnSchema("id", "INTEGER"),
                ColumnSchema("email", "STRING"),
                ColumnSchema("first_name", "STRING"),
                ColumnSchema("state", "STRING"),
                ColumnSchema("sale_price", "FLOAT"),
            ),
        )

    def describe_all(self):
        return [self.describe(t) for t in self.list_tables()]

    def dry_run(self, sql):
        return DryRunResult(bytes_processed=1_000)

    def assert_within_budget(self, sql):
        return self.dry_run(sql)

    def execute(self, sql):
        self.executed.append(sql)
        for marker in self.failing:
            if marker in sql:
                raise QuerySyntaxError(f"Syntax error near {marker}")
        if any(marker in sql for marker in self.empty_for):
            return QueryResult(rows=pd.DataFrame(), bytes_billed=1_000, row_count=0)
        if any(marker in sql for marker in self.null_aggregate_for):
            frame = pd.DataFrame({"total_revenue": [None]})
            return QueryResult(rows=frame, bytes_billed=1_000, row_count=1)
        frame = next(iter(self.frames.values()), pd.DataFrame())
        # `total_rows` lets a test say more rows matched than were fetched,
        # which is what the real warehouse reports when the read is capped.
        return QueryResult(
            rows=frame,
            bytes_billed=1_000,
            row_count=self.total_rows if self.total_rows is not None else len(frame),
        )


@pytest.fixture
def settings():
    return Settings(_env_file=None, google_cloud_project="test", repair_budget=2)


@pytest.fixture
def source():
    return FakeSource(
        frames={
            "default": pd.DataFrame(
                {"id": [1, 2], "email": ["a@b.com", "c@d.com"], "spend": [100, 90]}
            )
        }
    )


@pytest.fixture
def reports():
    return InMemoryReportStore()


@pytest.fixture
def traces():
    return InMemoryTraceStore()


@pytest.fixture
def make_deps(settings, source, reports, traces):
    def _make(script: list | None = None, src=None, store=None, **extra):
        return AgentDeps(
            settings=settings,
            llm=ScriptedChatModel(script or []),
            source=src or source,
            policy=PiiPolicy.default(),
            reports=store or reports,
            traces=traces,
            **extra,
        )

    return _make
