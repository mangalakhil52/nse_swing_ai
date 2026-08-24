"""#14R end-to-end provider wiring tests."""
import pytest
from src.runtime.end_to_end_scan import build_scan_orchestrator


def test_build_requires_both_live_provider_urls():
    with pytest.raises(RuntimeError, match="NSE_UNIVERSE_URL"):
        build_scan_orchestrator({}, lambda *a: None, lambda *a: None)
    with pytest.raises(RuntimeError, match="MARKET_DATA_BASE_URL"):
        build_scan_orchestrator({"universe_url": "https://u.test"}, lambda *a: None, lambda *a: None)


def test_build_constructs_real_provider_chain():
    orchestrator = build_scan_orchestrator({
        "universe_url": "https://u.test/universe.csv",
        "market_data_url": "https://m.test/data",
        "timeout_seconds": 5,
        "retries": 2,
        "max_workers": 3,
    }, lambda *a: None, lambda *a: None)
    assert orchestrator.max_workers == 3
    assert orchestrator.universe_service.adapter.source_url.endswith("universe.csv")
    assert orchestrator.market_data_adapter.retries == 2
