"""#14U controlled official-NSE runtime tests."""
from datetime import date
from unittest.mock import Mock
from src.runtime.official_nse_runtime import build_official_nse_orchestrator


def test_official_nse_runtime_builds_without_vendor_url_configuration():
    candidate = Mock()
    decision = Mock()
    orchestrator = build_official_nse_orchestrator(date(2026, 6, 30), candidate, decision, timeout_seconds=7, max_workers=2)
    assert orchestrator.max_workers == 2
    assert orchestrator.market_data_adapter.retries == 2
    assert orchestrator.universe_service.source_adapter.timeout_seconds == 7


def test_official_runtime_is_point_in_time_dated():
    orchestrator = build_official_nse_orchestrator(date(2026, 6, 30), Mock(), Mock())
    assert orchestrator.market_data_adapter.fetcher.__self__.as_of_date == date(2026, 6, 30)
