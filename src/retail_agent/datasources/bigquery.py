"""BigQuery adapter with a schema cache and a hard cost ceiling."""

from __future__ import annotations

import logging

from google.api_core import exceptions as gexc
from google.cloud import bigquery

from retail_agent.config import Settings
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
            completed = job.result(timeout=self._settings.bq_timeout_seconds)
            frame = completed.to_dataframe()
        except gexc.GoogleAPICallError as err:
            raise self._translate(err) from err

        return QueryResult(
            rows=frame,
            bytes_billed=int(getattr(job, "total_bytes_billed", 0) or 0),
            row_count=len(frame),
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
