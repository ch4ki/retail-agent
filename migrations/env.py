"""Alembic environment.

The database URL comes from `Settings` rather than `alembic.ini`, so there is
one source of truth for it and no credentials in a tracked file.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from retail_agent.config import Settings
from retail_agent.store.db import include_object, to_sqlalchemy_url
from retail_agent.store.models import Base

config = context.config

target_metadata = Base.metadata


def _database_url() -> str:
    override = config.get_main_option("sqlalchemy.url", "")
    if override and not override.startswith("driver://"):
        return override
    return to_sqlalchemy_url(Settings().database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
