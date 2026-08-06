"""Runs against a real Postgres. Deselected by default:

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db

The bodies come from the same contract the in-memory store satisfies. If the
two ever disagree, every component test that uses the double is testing a
fiction.
"""

import pytest

from retail_agent.config import Settings
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from retail_agent.store.reports import PostgresReportStore, ReportStore
from tests.support.report_store_contract import ReportStoreContract

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def sessions():
    settings = Settings()
    try:
        run_migrations(settings.database_url)
        engine = create_db_engine(settings.database_url)
        factory = session_factory(engine)
    except Exception as err:
        pytest.skip(f"Postgres unavailable: {err}")
    yield factory
    engine.dispose()


class TestPostgresReportStore(ReportStoreContract):
    @pytest.fixture
    def store(self, sessions):
        from sqlalchemy import text

        with sessions.begin() as session:
            session.execute(text("TRUNCATE reports, report_audit"))
        return PostgresReportStore(sessions)


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresReportStore(sessions), ReportStore)
