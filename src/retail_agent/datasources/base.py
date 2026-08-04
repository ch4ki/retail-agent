"""Warehouse-agnostic interface. Adding Snowflake means adding one adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


class DataSourceError(Exception):
    """Base for anything the warehouse rejects or fails on."""


class QuerySyntaxError(DataSourceError):
    """Query was malformed. Recoverable: the agent may rewrite and retry."""


class QueryCostError(DataSourceError):
    """Query would scan more than the configured budget. Not retried as-is."""


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    type: str
    mode: str = "NULLABLE"
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSchema, ...]

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def to_ddl(self) -> str:
        """Compact schema rendering for prompts."""
        lines = [f"{self.name}("]
        for column in self.columns:
            comment = f"  -- {column.description}" if column.description else ""
            lines.append(f"  {column.name} {column.type},{comment}")
        lines.append(")")
        return "\n".join(lines)


@dataclass(frozen=True)
class DryRunResult:
    bytes_processed: int

    @property
    def gigabytes(self) -> float:
        return self.bytes_processed / 1_000_000_000


@dataclass(frozen=True)
class QueryResult:
    rows: pd.DataFrame
    bytes_billed: int
    row_count: int


@runtime_checkable
class DataSource(Protocol):
    dialect: str

    def list_tables(self) -> list[str]: ...
    def describe(self, table: str) -> TableSchema: ...
    def describe_all(self) -> list[TableSchema]: ...
    def dry_run(self, sql: str) -> DryRunResult: ...
    def assert_within_budget(self, sql: str) -> DryRunResult: ...
    def execute(self, sql: str) -> QueryResult: ...
