"""The persona contract against a real Postgres.

    docker compose up -d postgres && uv run retail-agent migrate
    uv run pytest -m db
"""

import pytest

from retail_agent.config import Settings
from retail_agent.store.db import create_db_engine, run_migrations, session_factory
from retail_agent.store.personas import PersonaStore, PostgresPersonaStore
from tests.support.persona_store_contract import PersonaStoreContract

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


class TestPostgresPersonaStore(PersonaStoreContract):
    @pytest.fixture
    def store(self, sessions):
        from sqlalchemy import text

        with sessions.begin() as session:
            session.execute(text("TRUNCATE personas"))
        return PostgresPersonaStore(sessions)


def test_satisfies_the_protocol(sessions):
    assert isinstance(PostgresPersonaStore(sessions), PersonaStore)


def test_the_database_refuses_two_active_personas(sessions):
    """The single-active rule is a partial unique index, not a convention."""
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    with sessions.begin() as session:
        session.execute(text("TRUNCATE personas"))

    store = PostgresPersonaStore(sessions)
    store.save(name="a", body="A", updated_by="ceo")
    store.save(name="b", body="B", updated_by="ceo")
    store.activate(name="a")

    with pytest.raises(IntegrityError):
        with sessions.begin() as session:
            session.execute(
                text("UPDATE personas SET is_active = true WHERE name = 'b'")
            )
