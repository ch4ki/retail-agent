"""The trio contract against a real Postgres.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db
"""

import pytest

from retail_agent.config import Settings
from retail_agent.knowledge.trios import PostgresTrioStore, TrioStore
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from tests.support.trio_store_contract import TrioStoreContract

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


class TestPostgresTrioStore(TrioStoreContract):
    @pytest.fixture
    def store(self, sessions):
        """Clean before *and* after.

        Truncating only on setup left whichever test ran last with its rows
        still in the database — and this is the database the CLI reads, so a
        stray `id='a'` trio showed up in a live run as a real analyst
        definition.
        """
        from sqlalchemy import text

        def clear():
            with sessions.begin() as session:
                # CASCADE because trio_embeddings references trios.
                session.execute(text("TRUNCATE trios CASCADE"))

        clear()
        yield PostgresTrioStore(sessions)
        clear()


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresTrioStore(sessions), TrioStore)
