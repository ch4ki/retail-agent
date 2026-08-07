"""A scripted LLM and warehouse so graph behaviour can be tested offline."""

from dataclasses import dataclass, field

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

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
    def _make(replies: list, src=None, blocks: bool = False, store=None):
        return AgentDeps(
            settings=settings,
            llm=ScriptedLLM(replies, blocks=blocks),
            source=src or source,
            policy=PiiPolicy.default(),
            reports=store or reports,
            traces=traces,
        )

    return _make
