"""Database access: one engine, one session factory, and the migration entry point.

Schema changes are Alembic revisions under `migrations/`, not hand-applied SQL
files: they carry checksums, they can be rolled back, and `alembic history`
answers "what shape is this database in" without reading the code.

LangGraph's `PostgresSaver` manages its own `checkpoint_*` tables through its own
`setup()`. Alembic owns everything else in this database and does not touch them.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# LangGraph's PostgresSaver creates and migrates these itself. Alembic must not
# manage them, propose dropping them on autogenerate, or count them as drift.
# Defined here rather than in migrations/env.py because env.py can only be
# imported by Alembic — it reads `alembic.context` at module scope — and this
# rule needs to be readable from a test as well.
LANGGRAPH_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    return not (type_ == "table" and name in LANGGRAPH_TABLES)
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


@lru_cache(maxsize=None)
def shared_engine(database_url: str) -> Engine:
    """One engine per database URL per process.

    Seven stores talk to the same Postgres. Each building its own engine meant
    seven pools — up to seventy connections from a single CLI process — for a
    resource SQLAlchemy explicitly intends to be shared. `create_db_engine`
    stays uncached so tests can still make and dispose their own.
    """
    return create_db_engine(database_url)


def sessions_or_none(
    database_url: str, *, name: str, on_degraded=None, probe=None
) -> sessionmaker[Session] | None:
    """A session factory, or None when the database cannot be reached.

    Every store made this same decision in its own fifteen lines: connect,
    probe, log, call back, fall back to memory. The fallback itself differs per
    store — some seed, one wraps a cache — but the decision does not, and having
    it in one place is what stops the seven copies drifting.

    The probe is deliberate rather than lazy: a database that is up but never
    migrated should degrade at startup, not on the first write. `probe` takes a
    connection and runs a stronger check where a bare connect is not enough.
    """
    try:
        engine = shared_engine(database_url)
        with engine.connect() as conn:
            if probe is not None:
                probe(conn)
        return session_factory(engine)
    except Exception as err:
        logging.getLogger(__name__).debug("%s degraded: %s", name, err)
        if on_degraded is not None:
            on_degraded()
        return None


@lru_cache(maxsize=None)
def shared_engine(database_url: str) -> Engine:
    """One engine per URL per process.

    Seven stores each built their own, so a single CLI process could hold seven
    pools against the same database — and probed the connection seven times at
    startup. They want one pool between them.
    """
    return create_db_engine(database_url)


def sessions_or_none(
    database_url: str, *, name: str, on_degraded=None, probe=None
) -> sessionmaker[Session] | None:
    """A session factory, or None when the database is unreachable.

    Every store degrades the same way and each had its own copy of the same
    fifteen lines. The decision belongs in one place; what to do about it — seed
    from memory, warn the user, drop the feature — stays with the caller.
    """
    try:
        engine = shared_engine(database_url)
        with engine.connect():
            pass
        return session_factory(engine)
    except Exception as err:
        log.debug("%s degraded: %s", name, err)
        if on_degraded is not None:
            on_degraded()
        return None
