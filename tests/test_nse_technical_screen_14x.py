"""Stage-2 technical funnel contract tests."""
from datetime import date
from unittest.mock import patch
import pandas as pd

from src.runtime.nse_technical_screen import run


def _df():
    idx = pd.date_range("2026-01-02", periods=120, freq="B")
    close = pd.Series(range(100, 220), index=idx, dtype=float)
    return pd.DataFrame({
        "timestamp": idx,
        "open": close.values - 1,
        "high": close.values + 1,
        "low": close.values - 2,
        "close": close.values,
        "volume": [2_000_000.0] * len(idx),
    })


def test_stage2_only_analyzes_stage1_eligible_symbols():
    raw = [
        {"symbol": "AAA", "exchange": "NSE"},
        {"symbol": "BBB", "exchange": "NSE"},
    ]
    with patch("src.runtime.nse_technical_screen.NSEOfficialUniverseSource.fetch", return_value=raw), \
         patch("src.runtime.nse_technical_screen.NSEHistoricalOHLCVSource.fetch_many", return_value={"AAA": _df(), "BBB": _df()}):
        summary, rows = run(date(2026, 6, 30), max_workers=2)

    assert summary.universe_count == 2
    assert summary.stage1_eligible == 2
    assert summary.technical_analyzed == 2
    assert summary.technical_errors == 0
    assert all(row["pit_safe"] for row in rows)


def test_stage2_respects_limit():
    raw = [
        {"symbol": "AAA", "exchange": "NSE"},
        {"symbol": "BBB", "exchange": "NSE"},
    ]
    with patch("src.runtime.nse_technical_screen.NSEOfficialUniverseSource.fetch", return_value=raw), \
         patch("src.runtime.nse_technical_screen.NSEHistoricalOHLCVSource.fetch_many", return_value={"AAA": _df()}):
        summary, rows = run(date(2026, 6, 30), limit=1)

    assert summary.normalized_count == 1
    assert summary.stage1_eligible == 1
    assert len(rows) == 1
