"""Migrations are Alembic revisions now.

The URL helper is pure and always runs. The round-trip needs a real database
and is marked `db`.
"""

import pytest

from retail_agent.store.db import to_sqlalchemy_url


def test_plain_postgres_url_gains_the_psycopg_driver():
    """`DATABASE_URL` is shared with LangGraph's checkpointer, which wants libpq
    form. SQLAlchemy needs the driver named or it reaches for psycopg2."""
    assert (
        to_sqlalchemy_url("postgresql://retail:retail@localhost:5433/retail_agent")
        == "postgresql+psycopg://retail:retail@localhost:5433/retail_agent"
    )


def test_an_explicit_driver_is_left_alone():
    url = "postgresql+psycopg://retail@localhost/db"
    assert to_sqlalchemy_url(url) == url


@pytest.mark.db
def test_upgrade_and_downgrade_round_trip():
    """The hand-rolled runner this replaced was forward-only. A downgrade that
    has never been executed is a downgrade that does not work."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from retail_agent.config import Settings
    from retail_agent.store.db import ALEMBIC_INI, PROJECT_ROOT, create_db_engine

    settings = Settings()
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(settings.database_url))

    engine = create_db_engine(settings.database_url)
    try:
        command.downgrade(config, "base")
        assert "reports" not in inspect(engine).get_table_names()

        command.upgrade(config, "head")
        tables = inspect(engine).get_table_names()
        for table in ("reports", "report_audit", "traces", "turn_events"):
            assert table in tables, f"{table} missing after upgrade"
    finally:
        engine.dispose()


@pytest.mark.db
def test_migrations_match_the_models():
    """Revisions are hand-written when the database is not running, so nothing
    but this stops `models.py` and `migrations/` from drifting apart. Structural
    differences only — server defaults and index flavours produce noise that
    would make the test useless."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    from retail_agent.config import Settings
    from retail_agent.store.db import ALEMBIC_INI, PROJECT_ROOT, create_db_engine
    from retail_agent.store.models import Base

    settings = Settings()
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(settings.database_url))
    command.upgrade(config, "head")

    structural = {"add_table", "remove_table", "add_column", "remove_column"}
    engine = create_db_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()

    drifted = [d for d in diffs if isinstance(d, tuple) and d[0] in structural]
    assert not drifted, f"models and migrations disagree: {drifted}"
