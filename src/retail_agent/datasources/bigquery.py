"""BigQuery adapter with a schema cache and a hard cost ceiling."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from google.api_core import exceptions as gexc
from google.cloud import bigquery

from retail_agent.config import Settings
from retail_agent.datasources.column_values import (
    build_discovery_query,
    read_discovery_row,
)
from retail_agent.datasources.base import (
    ColumnSchema,
    DataSourceError,
    DryRunResult,
    QueryCostError,
    QueryResult,
    QuerySyntaxError,
    TableSchema,
)

log = logging.getLogger(__name__)

_SYNTAX_MARKERS = (
    "syntax error",
    "not found: field",
    "unrecognized name",
    "column not found",
    "invalid value",
)


class BigQuerySource:
    dialect = "bigquery"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self._client = client or bigquery.Client(project=settings.google_cloud_project)
        self._schema_cache: dict[str, TableSchema] = {}
        self._values_cache: dict[str, dict[str, tuple[str, ...]]] = {}

    # --- introspection ---

    def list_tables(self) -> list[str]:
        return sorted(self._settings.allowed_tables)

    def describe(self, table: str) -> TableSchema:
        if table in self._schema_cache:
            return self._schema_cache[table]

        try:
            meta = self._client.get_table(f"{self._settings.bq_dataset}.{table}")
        except gexc.NotFound as err:
            raise DataSourceError(f"Table '{table}' does not exist.") from err

        schema = TableSchema(
            name=table,
            columns=tuple(
                ColumnSchema(
                    name=field.name,
                    type=field.field_type,
                    mode=field.mode or "NULLABLE",
                    description=field.description or "",
                )
                for field in meta.schema
            ),
        )
        self._schema_cache[table] = schema
        return schema

    def describe_all(self) -> list[TableSchema]:
        return [self.describe(name) for name in self.list_tables()]

    def column_values(
        self, table: str, columns: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        """The values each low-cardinality column actually holds.

        Deliberately not routed through `execute`: that path is for the user's
        guarded, dry-run, cost-capped queries, and an internal metadata scan
        should not consume that budget or appear in the turn's SQL attempts.

        The caller decides which columns are safe to ask about — a PII column
        must never reach this method, because its values would land in a prompt.
        """
        cached = self._values_cache.get(table)
        if cached is not None:
            return cached

        sql = build_discovery_query(table, columns, dataset=self._settings.bq_dataset)
        if not sql:
            return {}

        job = self._client.query(sql)
        rows = list(job.result(timeout=self._settings.bq_timeout_seconds))
        found = read_discovery_row(dict(rows[0].items()), columns) if rows else {}
        # Cached here rather than by the caller: the SQL prompt is rebuilt on
        # every analysis turn, and these values do not change within a session.
        self._values_cache[table] = found
        return found

    # --- execution ---

    def dry_run(self, sql: str) -> DryRunResult:
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self._run(sql, config)
        return DryRunResult(bytes_processed=int(job.total_bytes_processed or 0))

    def assert_within_budget(self, sql: str) -> DryRunResult:
        estimate = self.dry_run(sql)
        if estimate.bytes_processed > self._settings.bq_max_bytes_billed:
            raise QueryCostError(
                f"Query would scan {estimate.gigabytes:.2f} GB, over the "
                f"{self._settings.bq_max_bytes_billed / 1e9:.2f} GB limit. "
                f"Add filters or narrow the columns."
            )
        return estimate

    def execute(self, sql: str) -> QueryResult:
        config = bigquery.QueryJobConfig(
            use_legacy_sql=False,
            use_query_cache=True,
            maximum_bytes_billed=self._settings.bq_max_bytes_billed,
        )
        job = self._run(sql, config)

        try:
            # Capped here rather than with a LIMIT in the SQL. A LIMIT truncates
            # server-side, so the size of the result is lost — 500 rows returned
            # looks the same whether 500 or 5,823 matched, and a query whose
            # rows were meant to be counted comes back silently wrong. Reading
            # with `max_results` bounds the transfer while `total_rows` still
            # reports the truth. It costs nothing: BigQuery bills bytes scanned,
            # and a LIMIT was measured to save 0% of them.
            completed = job.result(
                timeout=self._settings.bq_timeout_seconds,
                max_results=self._settings.display_row_limit,
            )
            frame = completed.to_dataframe()
        except gexc.GoogleAPICallError as err:
            raise self._translate(err) from err

        return QueryResult(
            rows=frame,
            bytes_billed=int(getattr(job, "total_bytes_billed", 0) or 0),
            # The real size of the result, not the number of rows fetched.
            row_count=int(completed.total_rows or len(frame)),
        )

    # --- internals ---

    def _run(self, sql: str, config: object):
        try:
            return self._client.query(sql, job_config=config)
        except gexc.GoogleAPICallError as err:
            raise self._translate(err) from err

    @staticmethod
    def _translate(err: Exception) -> DataSourceError:
        message = str(err)
        lowered = message.lower()

        if "exceeded" in lowered and "bytes billed" in lowered:
            return QueryCostError(message)
        if any(marker in lowered for marker in _SYNTAX_MARKERS):
            return QuerySyntaxError(message)
        if isinstance(err, gexc.BadRequest):
            return QuerySyntaxError(message)
        return DataSourceError(message)
