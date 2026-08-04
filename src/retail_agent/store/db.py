"""Postgres access: a shared pool and a minimal forward-only migration runner."""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg_pool import ConnectionPool

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def open_pool(database_url: str) -> ConnectionPool:
    return ConnectionPool(database_url, min_size=1, max_size=5, open=True)


def pending_migrations(directory: Path, applied: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return [path for path in sorted(directory.glob("*.sql")) if path.name not in applied]


def applied_migrations(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.schema_migrations') IS NOT NULL")
        (exists,) = cur.fetchone()
        if not exists:
            return set()
        cur.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def run_migrations(database_url: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every unapplied .sql file in name order. Returns what was applied."""
    executed: list[str] = []

    with ConnectionPool(database_url, min_size=1, max_size=1, open=True) as pool:
        with pool.connection() as conn:
            already = applied_migrations(conn)
            for path in pending_migrations(directory, already):
                log.info("applying migration %s", path.name)
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s) "
                    "ON CONFLICT DO NOTHING",
                    (path.name,),
                )
                conn.commit()
                executed.append(path.name)

    return executed
