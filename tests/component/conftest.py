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


class ScriptedLLM:
    """Returns queued replies in order. Records every prompt it received."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(_as_text(messages))
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return AIMessage(content=self.replies.pop(0))


def _as_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(getattr(m, "content", m)) for m in messages)


@dataclass
class FakeSource:
    """Serves canned frames; raises for SQL containing a marker in `failing`."""

    frames: dict[str, pd.DataFrame]
    failing: set[str] = field(default_factory=set)
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
        frame = next(iter(self.frames.values()), pd.DataFrame())
        return QueryResult(rows=frame, bytes_billed=1_000, row_count=len(frame))


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
def make_deps(settings, source):
    def _make(replies: list[str], src=None):
        return AgentDeps(
            settings=settings,
            llm=ScriptedLLM(replies),
            source=src or source,
            policy=PiiPolicy.default(),
        )

    return _make
