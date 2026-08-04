"""Live checks against real BigQuery. Deselected by default.

Run with: uv run pytest -m live
"""

import pytest

from retail_agent.config import Settings
from retail_agent.datasources.bigquery import BigQuerySource

pytestmark = pytest.mark.live


@pytest.fixture
def source():
    # GOOGLE_CLOUD_PROJECT is an explicit override. When it is unset the client
    # falls back to the project in application-default credentials, so skip on
    # a real connection failure rather than on a missing env var.
    try:
        return BigQuerySource(Settings())
    except Exception as err:
        pytest.skip(f"BigQuery unavailable: {err}")


def test_describe_users_returns_real_columns(source):
    schema = source.describe("users")
    assert "email" in schema.column_names()


def test_small_query_returns_a_dataframe(source):
    result = source.execute(
        "SELECT id, state FROM `bigquery-public-data.thelook_ecommerce.users` LIMIT 5"
    )
    assert result.row_count == 5
    assert list(result.rows.columns) == ["id", "state"]


def test_dry_run_reports_bytes(source):
    estimate = source.dry_run(
        "SELECT order_id FROM `bigquery-public-data.thelook_ecommerce.orders`"
    )
    assert estimate.bytes_processed > 0


def test_bad_column_is_classified_as_a_syntax_error(source):
    # This is the failure the repair edge keys off, so the classification of a
    # real BigQuery 400 matters more than the message text.
    from retail_agent.datasources.base import QuerySyntaxError

    with pytest.raises(QuerySyntaxError):
        source.execute(
            "SELECT nope FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 1"
        )
