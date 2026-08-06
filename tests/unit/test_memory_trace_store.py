import pytest

from retail_agent.obs.traces import InMemoryTraceStore, TraceStore
from tests.support.trace_store_contract import TraceStoreContract


class TestInMemoryTraceStore(TraceStoreContract):
    @pytest.fixture
    def store(self):
        return InMemoryTraceStore()


def test_satisfies_the_protocol():
    assert isinstance(InMemoryTraceStore(), TraceStore)
