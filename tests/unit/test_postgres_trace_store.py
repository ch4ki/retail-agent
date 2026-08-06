"""The trace contract against a real Postgres.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db
"""

import pytest

from retail_agent.config import Settings
from retail_agent.obs.traces import PostgresTraceStore, TraceStore
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from tests.support.trace_store_contract import TraceStoreContract

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def sessions():
    settings = Settings()
    try:
        run_migrations(settings.database_url)
        engine = create_db_engine(settings.database_url)
        factory = session_factory(engine)
        with engine.connect() as conn:
            conn.close()
    except Exception as err:
        pytest.skip(f"Postgres unavailable: {err}")
    yield factory
    engine.dispose()


class TestPostgresTraceStore(TraceStoreContract):
    @pytest.fixture
    def store(self, sessions):
        from sqlalchemy import text

        with sessions.begin() as session:
            session.execute(text("TRUNCATE traces, turn_events"))
        return PostgresTraceStore(sessions)


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresTraceStore(sessions), TraceStore)
