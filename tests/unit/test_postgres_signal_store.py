"""The signal contract against a real Postgres.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db

This is the implementation that makes the learning loop real: the threshold is
three, and evidence that dies with the process can only ever be met inside a
single session.
"""

import pytest

from retail_agent.config import Settings
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from retail_agent.store.learning import PostgresSignalStore, SignalStore
from tests.support.signal_store_contract import SignalStoreContract

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def sessions():
    settings = Settings()
    try:
        run_migrations(settings.database_url)
        engine = create_db_engine(settings.database_url)
    except Exception as err:
        pytest.skip(f"Postgres unavailable: {err}")
    yield session_factory(engine)
    engine.dispose()


class TestPostgresSignalStore(SignalStoreContract):
    @pytest.fixture
    def store(self, sessions):
        from sqlalchemy import text

        with sessions.begin() as session:
            session.execute(text("TRUNCATE preference_signals, preference_declines"))
        return PostgresSignalStore(sessions)


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresSignalStore(sessions), SignalStore)
