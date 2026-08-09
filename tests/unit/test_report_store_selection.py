from retail_agent.config import Settings
from retail_agent.store.reports import InMemoryReportStore, build_report_store

UNREACHABLE = "postgresql://nobody@127.0.0.1:1/nothing"


def test_falls_back_to_memory_when_postgres_is_unreachable():
    """Losing the database should cost report persistence, not the agent."""
    settings = Settings(_env_file=None, database_url=UNREACHABLE)

    assert isinstance(build_report_store(settings), InMemoryReportStore)


def test_the_caller_is_told_when_it_degrades():
    settings = Settings(_env_file=None, database_url=UNREACHABLE)
    told = []

    build_report_store(settings, on_degraded=lambda: told.append(True))

    assert told == [True]
