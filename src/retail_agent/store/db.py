"""Database access: one engine, one session factory, and the migration entry point.

Schema changes are Alembic revisions under `migrations/`, not hand-applied SQL
files: they carry checksums, they can be rolled back, and `alembic history`
answers "what shape is this database in" without reading the code.

LangGraph's `PostgresSaver` manages its own `checkpoint_*` tables through its own
`setup()`. Alembic owns everything else in this database and does not touch them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

# psycopg 3 rather than the psycopg2 default SQLAlchemy assumes.
_DRIVER_PREFIX = "postgresql+psycopg://"


def to_sqlalchemy_url(database_url: str) -> str:
    """Accept the plain `postgresql://` URL the rest of the project uses.

    `DATABASE_URL` is shared with LangGraph's checkpointer, which wants libpq
    form, so the driver suffix is added here rather than in the environment.
    """
    if database_url.startswith("postgresql+"):
        return database_url
    return database_url.replace("postgresql://", _DRIVER_PREFIX, 1)


def create_db_engine(database_url: str, *, connect_timeout: int = 2) -> Engine:
    """An engine with a bounded connect timeout.

    Without one, a dead host is discovered only after the driver's own retries,
    which is how a degraded startup turns into a minute of silence.
    """
    return create_engine(
        to_sqlalchemy_url(database_url),
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        connect_args={"connect_timeout": connect_timeout},
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """`expire_on_commit=False` so returned rows stay readable after commit —
    the store converts them to dataclasses and the session closes."""
    return sessionmaker(engine, expire_on_commit=False)


def run_migrations(database_url: str) -> str:
    """Upgrade the database to head. Returns the revision it landed on."""
    from alembic import command
    from alembic.config import Config

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(database_url))

    command.upgrade(config, "head")
    return current_revision(database_url)


def current_revision(database_url: str) -> str:
    from alembic.migration import MigrationContext

    engine = create_db_engine(database_url)
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision() or "base"
    finally:
        engine.dispose()
