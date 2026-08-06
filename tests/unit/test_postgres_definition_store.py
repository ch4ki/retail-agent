"""The definition contract against a real Postgres.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db
"""

import pytest

from retail_agent.config import Settings
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from retail_agent.store.definitions import DefinitionStore, PostgresDefinitionStore
from tests.support.definition_store_contract import DefinitionStoreContract

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


class TestPostgresDefinitionStore(DefinitionStoreContract):
    @pytest.fixture
    def store(self, sessions):
        from sqlalchemy import text

        with sessions.begin() as session:
            session.execute(text("TRUNCATE user_definitions"))
        return PostgresDefinitionStore(sessions)


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresDefinitionStore(sessions), DefinitionStore)
