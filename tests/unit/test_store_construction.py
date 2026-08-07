"""One engine per process, and one place that decides Postgres-or-memory.

Seven stores each built their own engine with its own pool — up to seventy
connections from a single CLI process, and seven separate connection probes at
startup — while running the same fifteen lines of try/connect/degrade.
"""

from __future__ import annotations

from retail_agent.store.db import shared_engine, sessions_or_none

URL = "postgresql://nobody@127.0.0.1:1/none"


def test_the_same_url_yields_the_same_engine():
    """Seven stores against one database want one pool, not seven."""
    assert shared_engine(URL) is shared_engine(URL)


def test_a_different_url_yields_a_different_engine():
    assert shared_engine(URL) is not shared_engine(URL.replace("/none", "/other"))


def test_an_unreachable_database_yields_no_sessions():
    """Every store degrades the same way, so the decision belongs in one place."""
    assert sessions_or_none(URL, name="test store") is None


def test_the_caller_is_told_when_it_degrades():
    told = []

    sessions_or_none(URL, name="test store", on_degraded=lambda: told.append(1))

    assert told == [1]


def test_no_callback_is_fine():
    assert sessions_or_none(URL, name="test store", on_degraded=None) is None
