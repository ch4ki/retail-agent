"""The preference contract against a real Postgres.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db
"""

import pytest

from retail_agent.config import Settings
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from retail_agent.store.preferences import PostgresPreferenceStore, PreferenceStore
from tests.support.preference_store_contract import PreferenceStoreContract

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


class TestPostgresPreferenceStore(PreferenceStoreContract):
    @pytest.fixture
    def store(self, sessions):
        from sqlalchemy import text

        with sessions.begin() as session:
            session.execute(text("TRUNCATE preferences"))
        return PostgresPreferenceStore(sessions)


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresPreferenceStore(sessions), PreferenceStore)
