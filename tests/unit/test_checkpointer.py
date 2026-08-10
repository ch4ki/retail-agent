"""Which database failures cost you history, and which are bugs.

The fallback to an in-memory saver is deliberate — a missing database should
cost history across restarts, not the ability to use the agent. What was wrong
was its breadth: `except Exception` also swallowed genuine startup bugs, and
reported every one of them as "Postgres unreachable".
"""

import io
from contextlib import contextmanager

import psycopg
import psycopg_pool
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from rich.console import Console

from retail_agent.cli import chat


class _FakeCursor:
    """Reports whatever migration version the test asks for."""

    def __init__(self, applied_version):
        self._applied_version = applied_version

    def execute(self, sql, params=None):
        return self

    def fetchone(self):
        return {"v": self._applied_version}


class _FakeSaver:
    """A saver whose probe behaves however the test needs.

    `MIGRATIONS` mirrors the real `PostgresSaver.MIGRATIONS` so `expected =
    len(saver.MIGRATIONS) - 1` inside `_assert_migrated` means the same thing
    here as it does against a real saver. `applied_version` defaults to the
    current one, so tests that do not care about the version check (most of
    them) get a saver that passes it.
    """

    MIGRATIONS = PostgresSaver.MIGRATIONS

    def __init__(self, on_probe=None, applied_version=None):
        self._on_probe = on_probe
        self._applied_version = (
            applied_version if applied_version is not None else len(self.MIGRATIONS) - 1
        )

    def get_tuple(self, config):
        if self._on_probe is not None:
            raise self._on_probe
        return None

    @contextmanager
    def _cursor(self):
        yield _FakeCursor(self._applied_version)


class _FakeConnection:
    def __init__(self, saver):
        self._saver = saver
        self.closed = False

    def __enter__(self):
        return self._saver

    def __exit__(self, *exc):
        self.closed = True
        return False


@pytest.fixture
def console():
    return Console(record=True, width=100, file=io.StringIO())


def _use(console, monkeypatch, *, opens=None, raises=None):
    """Run `_checkpointer` and report which saver it yielded."""

    def from_conn_string(url):
        if raises is not None:
            raise raises
        return opens

    monkeypatch.setattr(chat.PostgresSaver, "from_conn_string", from_conn_string)

    with chat._checkpointer(console, "postgresql://x/y") as saver:
        return saver


def test_a_healthy_database_yields_the_real_saver(console, monkeypatch):
    saver = _FakeSaver()

    assert _use(console, monkeypatch, opens=_FakeConnection(saver)) is saver


def test_an_unreachable_database_falls_back_with_a_warning(console, monkeypatch):
    yielded = _use(
        console, monkeypatch, raises=psycopg.OperationalError("connection refused")
    )

    assert isinstance(yielded, MemorySaver)
    assert "unreachable" in console.export_text().lower()


def test_a_pool_timeout_falls_back_too(console, monkeypatch):
    yielded = _use(console, monkeypatch, raises=psycopg_pool.PoolTimeout("timed out"))

    assert isinstance(yielded, MemorySaver)


def test_an_unmigrated_database_names_the_command_that_fixes_it(console, monkeypatch):
    """Reporting this as "unreachable" would send the reader to restart a
    database that is running fine."""
    connection = _FakeConnection(_FakeSaver(on_probe=psycopg.errors.UndefinedTable()))

    yielded = _use(console, monkeypatch, opens=connection)

    assert isinstance(yielded, MemorySaver)
    assert "retail-agent migrate" in console.export_text()
    assert connection.closed, "the connection was left open"


def test_a_database_at_the_current_migration_version_yields_the_real_saver(
    console, monkeypatch
):
    """The read probe passing is not enough on its own — this pins that a
    saver whose recorded version matches what the installed package expects
    is treated as healthy, not as behind."""
    saver = _FakeSaver(applied_version=len(PostgresSaver.MIGRATIONS) - 1)
    connection = _FakeConnection(saver)

    assert _use(console, monkeypatch, opens=connection) is saver


def test_an_older_migration_version_falls_back_naming_migrate(console, monkeypatch):
    """`SELECT_SQL` (what the read probe issues) does not reference every
    column a migration has added — `task_path`, added at migration index 9,
    is written but never read. So a database sitting one migration behind
    would pass the read probe and only fail once the agent tries to write.
    This pins that the version check catches it instead, at startup."""
    saver = _FakeSaver(applied_version=len(PostgresSaver.MIGRATIONS) - 2)
    connection = _FakeConnection(saver)

    yielded = _use(console, monkeypatch, opens=connection)

    assert isinstance(yielded, MemorySaver)
    assert "retail-agent migrate" in console.export_text()
    assert connection.closed, "the connection was left open"


def test_an_unexpected_failure_propagates(console, monkeypatch):
    """The whole point of narrowing the catch. A NameError in startup must not
    read as a database problem."""
    with pytest.raises(NameError):
        _use(console, monkeypatch, raises=NameError("build_persona_store"))


def test_a_repl_error_is_not_mistaken_for_a_database_error(console, monkeypatch):
    """`yield` used to sit inside the `try`, so an exception raised by the REPL
    body was thrown back in at the yield point and caught as a database
    failure."""
    monkeypatch.setattr(
        chat.PostgresSaver,
        "from_conn_string",
        lambda url: _FakeConnection(_FakeSaver()),
    )

    with pytest.raises(ZeroDivisionError):
        with chat._checkpointer(console, "postgresql://x/y"):
            1 / 0


@pytest.mark.db
def test_a_real_unmigrated_database_raises_undefined_table():
    """Pins the exception the unit tests above are written against.

    Against a database created fresh and never set up, the probe must raise
    exactly what `_checkpointer` catches. If LangGraph changes what it raises,
    this fails and the unit tests keep passing against a fiction.
    """
    import uuid

    from langgraph.checkpoint.postgres import PostgresSaver

    from retail_agent.cli.app import setup_checkpoint_tables
    from retail_agent.config import Settings

    admin_url = Settings(_env_file=None).database_url
    fresh = f"probe_{uuid.uuid4().hex[:8]}"

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{fresh}"')

    fresh_url = admin_url.rsplit("/", 1)[0] + f"/{fresh}"
    try:
        with PostgresSaver.from_conn_string(fresh_url) as saver:
            with pytest.raises(psycopg.errors.UndefinedTable):
                chat._assert_migrated(saver)

        setup_checkpoint_tables(fresh_url)

        with psycopg.connect(fresh_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT max(v) FROM checkpoint_migrations")
            applied = cur.fetchone()[0]
        assert applied == len(PostgresSaver.MIGRATIONS) - 1, (
            "setup() did not apply every migration the installed package has"
        )

        with PostgresSaver.from_conn_string(fresh_url) as saver:
            assert chat._assert_migrated(saver) is None, "probe fails after migrate"
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{fresh}" WITH (FORCE)')
