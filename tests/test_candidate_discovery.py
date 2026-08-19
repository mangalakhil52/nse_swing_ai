"""
P0 #14B — Candidate Discovery Engine Unit & Integration Tests.

Validates that:
  1. Candidate Discovery accepts any input universe (NO hardcoded symbols like RELIANCE/TCS).
  2. Multi-stock synthetic universes are screened deterministically.
  3. Historical & Live modes share identical screening logic using explicit as_of_date.
  4. Point-in-time safety is strictly enforced (future rows > as_of_date are sliced and cannot affect candidacy at T).
  5. Mutating future prices (> T) cannot change candidacy at T.
  6. Insufficient history (< min_history_length) fails with machine-readable reason INSUFFICIENT_HISTORY.
  7. Illiquid stocks fail with machine-readable reason INSUFFICIENT_LIQUIDITY.
  8. DataQualityGate failures (invalid geometry, PIT violation) hard-fail candidate eligibility.
  9. Rejection outputs are fully explainable (failed_filters and reasons populated).
 10. Historical universe unavailable raises HistoricalUniverseUnavailableError (no silent fallback to current watchlist).
"""

from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from src.candidate_discovery import (
    CandidateDiscoveryConfig,
    CandidateDiscoveryEngine,
    CandidateDiscoveryResult,
)
from src.core.models import SymbolMetadata
from src.data.historical_universe import HistoricalUniverseProvider, HistoricalUniverseUnavailableError


def _generate_ohlcv_df(
    symbol: str,
    num_bars: int = 60,
    end_date_str: str = "2026-06-30",
    base_price: float = 100.0,
    base_volume: int = 100000,
) -> pd.DataFrame:
    """Generates synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range(end=end_date_str, periods=num_bars, freq="B")
    np.random.seed(hash(symbol) % 10000)
    prices = base_price + np.cumsum(np.random.normal(0, 0.2, num_bars))
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": base_volume,
        "turnover_crores": (prices * base_volume) / 1e7,
    })


def test_multi_stock_universe():
    """1. Test screening multi-stock synthetic universe with different parameters."""
    as_of = date(2026, 6, 30)
    market_data = {
        "AAA": _generate_ohlcv_df("AAA", num_bars=60, base_price=200.0, base_volume=100000),  # Valid & Liquid
        "BBB": _generate_ohlcv_df("BBB", num_bars=60, base_price=10.0, base_volume=100000),   # Price < ₹20
        "CCC": _generate_ohlcv_df("CCC", num_bars=20, base_price=300.0, base_volume=100000),  # Insufficient history
        "DDD": _generate_ohlcv_df("DDD", num_bars=60, base_price=150.0, base_volume=100),     # Illiquid
    }

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["AAA", "BBB", "CCC", "DDD"],
        as_of_date=as_of,
        market_data_map=market_data,
        config=CandidateDiscoveryConfig(min_price=20.0, min_history_length=50, min_average_volume=10000),
    )

    res_map = {r.symbol: r for r in results}

    assert res_map["AAA"].eligible is True
    assert res_map["BBB"].eligible is False
    assert "PRICE_BELOW_MINIMUM" in res_map["BBB"].reasons

    assert res_map["CCC"].eligible is False
    assert "INSUFFICIENT_HISTORY" in res_map["CCC"].reasons

    assert res_map["DDD"].eligible is False
    assert "INSUFFICIENT_LIQUIDITY" in res_map["DDD"].reasons


def test_no_hardcoded_symbols():
    """2. Test Candidate Discovery accepts any arbitrary synthetic symbol (XYZ1, XYZ2)."""
    as_of = date(2026, 6, 30)
    market_data = {
        "XYZ1": _generate_ohlcv_df("XYZ1", num_bars=60, base_price=500.0, base_volume=50000),
        "XYZ2": _generate_ohlcv_df("XYZ2", num_bars=60, base_price=750.0, base_volume=60000),
    }

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["XYZ1", "XYZ2"],
        as_of_date=as_of,
        market_data_map=market_data,
    )

    symbols = [r.symbol for r in results if r.eligible]
    assert "XYZ1" in symbols
    assert "XYZ2" in symbols


def test_historical_pit():
    """3. Test dataset extending past as_of_date is sliced without consuming future rows."""
    as_of = date(2026, 6, 15)
    # DataFrame extends to June 30 > June 15
    df_raw = _generate_ohlcv_df("SYNTH1", num_bars=100, end_date_str="2026-06-30")

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["SYNTH1"],
        as_of_date=as_of,
        market_data_map={"SYNTH1": df_raw},
    )

    res = results[0]
    assert res.eligible is True
    assert res.pit_safe is True


def test_future_price_cannot_change_candidacy():
    """4. Test mutating future prices (> T) cannot change candidacy at T."""
    as_of = date(2026, 6, 15)
    df1 = _generate_ohlcv_df("MUTATE_TEST", num_bars=60, end_date_str="2026-06-30")

    results1 = CandidateDiscoveryEngine.discover_candidates(
        universe=["MUTATE_TEST"],
        as_of_date=as_of,
        market_data_map={"MUTATE_TEST": df1},
    )

    # Mutate ONLY future rows (> June 15)
    df2 = df1.copy()
    future_mask = df2["timestamp"] > pd.to_datetime(as_of)
    df2.loc[future_mask, "close"] = 999999.0
    df2.loc[future_mask, "high"] = 999999.0

    results2 = CandidateDiscoveryEngine.discover_candidates(
        universe=["MUTATE_TEST"],
        as_of_date=as_of,
        market_data_map={"MUTATE_TEST": df2},
    )

    assert results1[0].eligible == results2[0].eligible
    assert results1[0].passed_filters == results2[0].passed_filters


def test_insufficient_history():
    """5. Test stock with history < min_history_length fails INSUFFICIENT_HISTORY."""
    as_of = date(2026, 6, 30)
    df_short = _generate_ohlcv_df("SHORT_HIST", num_bars=25)  # 25 bars < 50

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["SHORT_HIST"],
        as_of_date=as_of,
        market_data_map={"SHORT_HIST": df_short},
        config=CandidateDiscoveryConfig(min_history_length=50),
    )

    res = results[0]
    assert res.eligible is False
    assert "HISTORY_SUFFICIENCY" in res.failed_filters
    assert "INSUFFICIENT_HISTORY" in res.reasons


def test_liquidity():
    """6. Test illiquid stock fails with INSUFFICIENT_LIQUIDITY."""
    as_of = date(2026, 6, 30)
    df_illiquid = _generate_ohlcv_df("ILLIQUID", num_bars=60, base_volume=100)  # Vol 100

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["ILLIQUID"],
        as_of_date=as_of,
        market_data_map={"ILLIQUID": df_illiquid},
        config=CandidateDiscoveryConfig(min_average_volume=10000),
    )

    res = results[0]
    assert res.eligible is False
    assert "LIQUIDITY" in res.failed_filters
    assert "INSUFFICIENT_LIQUIDITY" in res.reasons


def test_data_quality():
    """7. Test DataQualityGate failure (broken OHLC geometry) rejects candidate."""
    as_of = date(2026, 6, 30)
    df_bad = _generate_ohlcv_df("BAD_GEOM", num_bars=60)
    # Corrupt bar: High (50) < Low (100)
    df_bad.loc[30, "high"] = 50.0
    df_bad.loc[30, "low"] = 100.0

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["BAD_GEOM"],
        as_of_date=as_of,
        market_data_map={"BAD_GEOM": df_bad},
    )

    res = results[0]
    assert res.eligible is False
    assert "DATA_QUALITY" in res.failed_filters
    assert "DATA_QUALITY_INVALID" in res.reasons


def test_explanation():
    """8. Test every rejected stock has failed_filters and reasons populated."""
    as_of = date(2026, 6, 30)
    df_penny = _generate_ohlcv_df("PENNY", num_bars=60, base_price=5.0)

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["PENNY"],
        as_of_date=as_of,
        market_data_map={"PENNY": df_penny},
        config=CandidateDiscoveryConfig(min_price=20.0),
    )

    res = results[0]
    assert res.eligible is False
    assert len(res.failed_filters) > 0
    assert len(res.reasons) > 0
    assert "PRICE_BELOW_MINIMUM" in res.reasons


def test_historical_universe_fail_closed():
    """9. Test unsupplied historical universe query fails closed with HistoricalUniverseUnavailableError."""
    with pytest.raises(HistoricalUniverseUnavailableError):
        HistoricalUniverseProvider.get_universe_for_date(date(2025, 6, 1))


def test_live_mode():
    """10. Test live mode uses identical screening logic for live watchlist."""
    as_of = date(2026, 8, 19)
    market_data = {
        "LIVE1": _generate_ohlcv_df("LIVE1", num_bars=60, end_date_str="2026-08-19"),
    }

    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["LIVE1"],
        as_of_date=as_of,
        market_data_map=market_data,
    )

    assert len(results) == 1
    assert results[0].eligible is True
