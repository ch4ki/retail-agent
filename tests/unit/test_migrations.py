from pathlib import Path

from retail_agent.store.db import MIGRATIONS_DIR, pending_migrations

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def test_module_points_at_the_real_migrations_directory():
    assert MIGRATIONS_DIR == MIGRATIONS
    assert (MIGRATIONS_DIR / "001_init.sql").is_file()


def test_discovers_migration_files_in_order():
    pending = pending_migrations(MIGRATIONS, applied=set())
    assert [p.name for p in pending] == sorted(p.name for p in pending)
    assert "001_init.sql" in [p.name for p in pending]


def test_already_applied_migrations_are_skipped():
    pending = pending_migrations(MIGRATIONS, applied={"001_init.sql"})
    assert "001_init.sql" not in [p.name for p in pending]


def test_missing_directory_yields_nothing(tmp_path):
    assert pending_migrations(tmp_path / "nope", applied=set()) == []
