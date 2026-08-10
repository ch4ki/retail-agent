"""`retail-agent migrate` owns every table the app needs.

LangGraph's checkpoint tables used to be created by `PostgresSaver.setup()` on
every `retail-agent chat` launch. Schema creation belongs with the other schema
creation, and running DDL on every launch is the thing the persistence guidance
names outright.

They are not folded into an Alembic revision: that would copy LangGraph's DDL
into this repository, where it would drift silently on the next upgrade.
"""

import io

import pytest
from rich.console import Console

from retail_agent.cli import app


@pytest.fixture
def console():
    return Console(record=True, width=100, file=io.StringIO())


def test_migrate_creates_the_checkpoint_tables(monkeypatch, console):
    monkeypatch.setattr(app, "run_migrations", lambda url: "abc123")

    urls = []
    monkeypatch.setattr(app, "setup_checkpoint_tables", urls.append)

    code = app._migrate(console=console)

    assert code == 0
    assert len(urls) == 1, "the checkpoint tables were not set up"


def test_migrate_reports_the_revision(monkeypatch, console):
    monkeypatch.setattr(app, "run_migrations", lambda url: "abc123")
    monkeypatch.setattr(app, "setup_checkpoint_tables", lambda url: None)

    app._migrate(console=console)

    assert "abc123" in console.export_text()


def test_a_checkpoint_setup_failure_is_reported_not_swallowed(monkeypatch, console):
    """A migrate that half-succeeded must not exit 0. The next `chat` would
    fall back to an in-memory saver and the reason would be invisible."""
    monkeypatch.setattr(app, "run_migrations", lambda url: "abc123")

    def explode(url):
        raise RuntimeError("no permission to create tables")

    monkeypatch.setattr(app, "setup_checkpoint_tables", explode)

    code = app._migrate(console=console)

    assert code == 1
    assert "no permission to create tables" in console.export_text()
