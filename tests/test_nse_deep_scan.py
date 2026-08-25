from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.runtime.nse_deep_scan import run


def _df(symbol="AAA"):
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1_000_000.0})


class _Hist:
    async def get_daily_ohlcv(self, *args, **kwargs):
        return _df("NIFTY 50")

    async def close(self):
        return None


def _regime(allow=True):
    return SimpleNamespace(allow_long_swing_trades=allow, regime=SimpleNamespace(value="BULL" if allow else "BEAR"), trading_stance=SimpleNamespace(value="NORMAL" if allow else "DEFENSIVE"), risk_multiplier=1.0)


def test_deep_scan_is_gated_by_market_regime():
    technical = [SimpleNamespace(symbol="AAA")]
    swing = SimpleNamespace(historical_diagnostics={}, recent_listing_shortlist=[])
    universe = [{"symbol": "AAA", "exchange": "NSE"}]
    with patch("src.runtime.nse_deep_scan.run_swing_scan", return_value=(swing, technical)), patch("src.runtime.nse_deep_scan.NSEOfficialUniverseSource.fetch", return_value=universe), patch("src.runtime.nse_deep_scan.NSEHistoricalOHLCVSource.fetch_many", return_value={"AAA": _df()}), patch("src.runtime.nse_deep_scan.HistoricalDataProvider", return_value=_Hist()), patch("src.runtime.nse_deep_scan.MarketRegimeClassifier.classify_regime", return_value=_regime(False)):
        summary, recs = run(date(2026, 8, 24))
    assert summary.technical_shortlist_count == 1
    assert summary.deep_candidates_count == 0
    assert summary.recommendations_count == 0
    assert recs == []


def test_deep_scan_only_sends_intelligence_gate_to_cio():
    technical = [SimpleNamespace(symbol="AAA")]
    swing = SimpleNamespace(historical_diagnostics={}, recent_listing_shortlist=[])
    universe = [{"symbol": "AAA", "exchange": "NSE"}]
    gated = SimpleNamespace(symbol="AAA")
    fake_rec = SimpleNamespace(symbol="AAA")
    with patch("src.runtime.nse_deep_scan.run_swing_scan", return_value=(swing, technical)), patch("src.runtime.nse_deep_scan.NSEOfficialUniverseSource.fetch", return_value=universe), patch("src.runtime.nse_deep_scan.NSEHistoricalOHLCVSource.fetch_many", return_value={"AAA": _df()}), patch("src.runtime.nse_deep_scan.HistoricalDataProvider", return_value=_Hist()), patch("src.runtime.nse_deep_scan.MarketRegimeClassifier.classify_regime", return_value=_regime(True)), patch("src.runtime.nse_deep_scan.select_normal", return_value=[gated]), patch("src.runtime.nse_deep_scan.select_recent", return_value=[]), patch("src.runtime.nse_deep_scan._run_cio", return_value=[fake_rec]) as cio:
        summary, recs = run(date(2026, 8, 24))
    assert summary.intelligence_normal_count == 1
    assert summary.intelligence_recent_count == 0
    assert summary.deep_candidates_count == 1
    assert recs == [fake_rec]
    assert cio.call_args.args[0][0] is gated


def test_deep_scan_can_route_recent_listing_track_to_cio():
    technical = []
    swing = SimpleNamespace(historical_diagnostics={}, recent_listing_shortlist=[{"symbol": "NEWCO"}])
    universe = [{"symbol": "NEWCO", "exchange": "NSE"}]
    gated = SimpleNamespace(symbol="NEWCO")
    fake_rec = SimpleNamespace(symbol="NEWCO")
    with patch("src.runtime.nse_deep_scan.run_swing_scan", return_value=(swing, technical)), patch("src.runtime.nse_deep_scan.NSEOfficialUniverseSource.fetch", return_value=universe), patch("src.runtime.nse_deep_scan.NSEHistoricalOHLCVSource.fetch_many", return_value={"NEWCO": _df("NEWCO")}), patch("src.runtime.nse_deep_scan.HistoricalDataProvider", return_value=_Hist()), patch("src.runtime.nse_deep_scan.MarketRegimeClassifier.classify_regime", return_value=_regime(True)), patch("src.runtime.nse_deep_scan.select_normal", return_value=[]), patch("src.runtime.nse_deep_scan.select_recent", return_value=[gated]), patch("src.runtime.nse_deep_scan._run_cio", return_value=[fake_rec]) as cio:
        summary, recs = run(date(2026, 8, 24))
    assert summary.intelligence_normal_count == 0
    assert summary.intelligence_recent_count == 1
    assert summary.deep_candidates_count == 1
    assert recs == [fake_rec]
    assert cio.call_args.args[0][0] is gated
