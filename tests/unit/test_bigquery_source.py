import pandas as pd
import pytest

from retail_agent.config import Settings
from retail_agent.datasources.base import DataSource, QueryCostError, QuerySyntaxError
from retail_agent.datasources.bigquery import (
    MAX_DISTINCT,
    BigQuerySource,
    build_discovery_query,
    read_discovery_row,
)


class FakeField:
    def __init__(self, name, field_type="STRING", mode="NULLABLE", description=""):
        self.name = name
        self.field_type = field_type
        self.mode = mode
        self.description = description


class FakeTable:
    def __init__(self, schema):
        self.schema = schema


class FakeJob:
    def __init__(self, df=None, total_bytes_processed=1000, error=None, total_rows=None):
        self._df = df if df is not None else pd.DataFrame({"id": [1]})
        self.total_bytes_processed = total_bytes_processed
        self.total_bytes_billed = total_bytes_processed
        self._error = error
        # What the warehouse says matched, which can exceed what is fetched.
        self.total_rows = total_rows if total_rows is not None else len(self._df)
        self.max_results = None

    def result(self, timeout=None, max_results=None):
        if self._error:
            raise self._error
        self.max_results = max_results
        return self

    def to_dataframe(self):
        return self._df

    def __iter__(self):
        # Real row objects expose .items(); dicts do too, which is all the
        # value-discovery path reads.
        return iter(self._df.to_dict("records"))


class FakeClient:
    def __init__(self, job=None, table=None):
        self._job = job or FakeJob()
        self._table = table or FakeTable([FakeField("id", "INTEGER")])
        self.last_config = None

    def query(self, sql, job_config=None):
        self.last_config = job_config
        return self._job

    def get_table(self, ref):
        return self._table


@pytest.fixture
def settings():
    return Settings(_env_file=None, google_cloud_project="test-project")


def test_conforms_to_datasource_protocol(settings):
    source = BigQuerySource(settings, client=FakeClient())
    assert isinstance(source, DataSource)
    assert source.dialect == "bigquery"


def test_describe_maps_bigquery_schema(settings):
    table = FakeTable(
        [FakeField("id", "INTEGER", "REQUIRED", "primary key"), FakeField("email")]
    )
    source = BigQuerySource(settings, client=FakeClient(table=table))

    schema = source.describe("users")

    assert schema.name == "users"
    assert schema.column_names() == ("id", "email")
    assert schema.columns[0].type == "INTEGER"
    assert "primary key" in schema.to_ddl()


def test_describe_caches_by_table(settings):
    client = FakeClient()
    calls = []
    original = client.get_table
    client.get_table = lambda ref: (calls.append(ref), original(ref))[1]

    source = BigQuerySource(settings, client=client)
    source.describe("users")
    source.describe("users")

    assert len(calls) == 1


def test_execute_applies_cost_cap(settings):
    client = FakeClient()
    source = BigQuerySource(settings, client=client)

    source.execute("SELECT id FROM users LIMIT 1")

    assert client.last_config.maximum_bytes_billed == settings.bq_max_bytes_billed
    assert client.last_config.use_legacy_sql is False


def test_execute_returns_rows_and_byte_count(settings):
    df = pd.DataFrame({"id": [1, 2, 3]})
    source = BigQuerySource(settings, client=FakeClient(FakeJob(df, 4096)))

    result = source.execute("SELECT id FROM users LIMIT 3")

    assert result.row_count == 3
    assert result.bytes_billed == 4096


def test_syntax_error_is_wrapped(settings):
    from google.api_core.exceptions import BadRequest

    job = FakeJob(error=BadRequest("Syntax error: Unexpected keyword FRM"))
    source = BigQuerySource(settings, client=FakeClient(job))

    with pytest.raises(QuerySyntaxError) as excinfo:
        source.execute("SELCT id FRM users")

    assert "Syntax error" in str(excinfo.value)


def test_over_budget_dry_run_raises_cost_error(settings):
    settings = settings.model_copy(update={"bq_max_bytes_billed": 100})
    source = BigQuerySource(
        settings, client=FakeClient(FakeJob(total_bytes_processed=999))
    )

    with pytest.raises(QueryCostError):
        source.assert_within_budget("SELECT id FROM order_items")


def test_within_budget_dry_run_returns_the_estimate(settings):
    source = BigQuerySource(
        settings, client=FakeClient(FakeJob(total_bytes_processed=2048))
    )

    estimate = source.assert_within_budget("SELECT id FROM users")

    assert estimate.bytes_processed == 2048


def test_list_tables_returns_allowed_tables(settings):
    source = BigQuerySource(settings, client=FakeClient())
    assert "orders" in source.list_tables()


def test_describe_all_covers_every_allowed_table(settings):
    source = BigQuerySource(settings, client=FakeClient())
    assert len(source.describe_all()) == len(settings.allowed_tables)


# --- reading a capped result without losing its true size ---


def test_the_fetch_is_capped_at_the_display_limit():
    """Not a LIMIT in the SQL. A LIMIT truncates server-side and the true size
    of the result is lost with it — and it saves no money, since BigQuery bills
    bytes scanned (measured at 0% on the query that exposed this)."""
    job = FakeJob(df=pd.DataFrame({"id": list(range(500))}), total_rows=5823)
    source = BigQuerySource(Settings(_env_file=None, google_cloud_project="p"), client=FakeClient(job))

    source.execute("SELECT id FROM users")

    assert job.max_results == 500


def test_the_row_count_is_the_true_total_not_the_number_fetched():
    """The whole point of capping at read time: "how many loyal customers" has
    a correct answer available even when the agent returned rows rather than a
    COUNT."""
    job = FakeJob(df=pd.DataFrame({"id": list(range(500))}), total_rows=5823)
    source = BigQuerySource(Settings(_env_file=None, google_cloud_project="p"), client=FakeClient(job))

    result = source.execute("SELECT id FROM users")

    assert result.row_count == 5823
    assert len(result.rows) == 500


def test_column_values_are_read_once_per_table():
    """The SQL prompt is rebuilt on every analysis turn. Paying for a warehouse
    scan each time would be a recurring cost for a fact that does not change
    within a session. Cached beside the schema cache, which already works this
    way — the caller used to do it by writing a private attribute onto here."""
    calls = []

    class CountingClient(FakeClient):
        def query(self, sql, job_config=None):
            calls.append(sql)
            return super().query(sql, job_config)

    job = FakeJob(df=pd.DataFrame([{"gender": [{"value": "F"}]}]))
    source = BigQuerySource(
        Settings(_env_file=None, google_cloud_project="p"), client=CountingClient(job)
    )

    first = source.column_values("users", ("gender",))
    second = source.column_values("users", ("gender",))

    assert first == second == {"gender": ("F",)}
    assert len(calls) == 1, f"queried {len(calls)} times for one table"


# --- value discovery ---
#
# Which columns are safe to ask about is warehouse-agnostic policy, tested in
# `test_column_values.py`. Asking is BigQuery SQL, so it is tested here.


def entries(*values):
    """What `APPROX_TOP_COUNT` hands back: one struct per distinct value."""
    return [{"value": value, "count": 1} for value in values]


def test_the_discovery_query_asks_for_the_values_of_every_candidate_column():
    sql = build_discovery_query("users", ("gender", "country"), dataset="ds")

    assert f"APPROX_TOP_COUNT(`gender`, {MAX_DISTINCT + 2})" in sql
    assert f"APPROX_TOP_COUNT(`country`, {MAX_DISTINCT + 2})" in sql
    assert "`ds.users`" in sql


def test_the_values_alone_say_whether_the_column_is_an_enumeration():
    """One aggregate per column, not two: the number of values that come back
    is the cardinality answer, so a separate count would be a second thing to
    keep in step with the first."""
    assert "APPROX_COUNT_DISTINCT" not in build_discovery_query(
        "users", ("gender",), dataset="ds"
    )


def test_no_columns_means_no_query():
    """A table of nothing but ids and timestamps must not be queried at all."""
    assert build_discovery_query("users", (), dataset="ds") == ""


def test_a_low_cardinality_column_yields_its_values():
    row = {"gender": entries("F", "M")}

    assert read_discovery_row(row, ("gender",)) == {"gender": ("F", "M")}


def test_a_high_cardinality_column_is_dropped():
    """230 states is more prompt than the mistakes it would prevent. The query
    asks for more slots than the ceiling precisely so that "over" is visible
    here rather than truncated into looking small enough."""
    row = {"state": entries(*(f"state-{i}" for i in range(MAX_DISTINCT + 1)))}

    assert read_discovery_row(row, ("state",)) == {}


def test_a_column_at_the_ceiling_survives_a_null_taking_a_slot():
    """Why the query asks for two slots more than the ceiling rather than one:
    NULL occupies a slot without being a value, so a column holding exactly
    `MAX_DISTINCT` values would otherwise look like one over."""
    values = tuple(f"value-{i:02d}" for i in range(MAX_DISTINCT))
    row = {"s": entries(*values, None)}

    assert read_discovery_row(row, ("s",)) == {"s": values}


def test_values_come_back_in_a_stable_order():
    """The schema goes into every prompt. Reordering it between runs would
    invalidate prompt caches and make two runs hard to diff."""
    row = {"s": entries("Shipped", "Complete", "Cancelled")}

    assert read_discovery_row(row, ("s",)) == {"s": ("Cancelled", "Complete", "Shipped")}


def test_nulls_are_not_offered_as_a_literal():
    """`WHERE status = 'None'` matches nothing. A NULL is not a value the model
    should be told to compare against."""
    row = {"s": entries("F", None)}

    assert read_discovery_row(row, ("s",)) == {"s": ("F",)}


def test_a_column_missing_from_the_row_is_skipped_not_an_error():
    assert read_discovery_row({}, ("gender",)) == {}
