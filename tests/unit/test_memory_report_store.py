import pytest

from retail_agent.store.memory_reports import InMemoryReportStore
from retail_agent.store.reports import ReportStore
from tests.support.report_store_contract import ReportStoreContract


class TestInMemoryReportStore(ReportStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryReportStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryReportStore(), ReportStore)
