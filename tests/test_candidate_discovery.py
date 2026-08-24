"""P0 #14B — Candidate Discovery Engine tests."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.candidate_discovery import CandidateDiscoveryConfig, CandidateDiscoveryEngine
from src.core.models import SymbolMetadata
from src.data.historical_universe import HistoricalUniverseProvider, HistoricalUniverseUnavailableError


def _generate_ohlcv_df(
    symbol: str,
    num_bars: int = 60,
    end_date_str: str = "2026-06-30",
    base_price: float = 100.0,
    base_volume: int = 100000,
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV fixture."""
    dates = pd.date_range(end=end_date_str, periods=num_bars, freq="B")
    rng = np.random.default_rng(sum(ord(c) for c in symbol))
    prices = base_price + np.cumsum(rng.normal(0, 0.2, num_bars))
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": base_volume,
            "turnover_crores": (prices * base_volume) / 1e7,
        }
    )


def test_multi_stock_universe():
    as_of = date(2026, 6, 30)
    market_data = {
        "AAA": _generate_ohlcv_df("AAA", 60, base_price=200.0, base_volume=100000),
        "BBB": _generate_ohlcv_df("BBB", 60, base_price=10.0, base_volume=100000),
        "CCC": _generate_ohlcv_df("CCC", 20, base_price=300.0, base_volume=100000),
        "DDD": _generate_ohlcv_df("DDD", 60, base_price=150.0, base_volume=100),
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
    as_of = date(2026, 6, 30)
    market_data = {
        "XYZ1": _generate_ohlcv_df("XYZ1", 60, base_price=500.0, base_volume=50000),
        "XYZ2": _generate_ohlcv_df("XYZ2", 60, base_price=750.0, base_volume=60000),
    }
    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["XYZ1", "XYZ2"], as_of_date=as_of, market_data_map=market_data
    )
    assert {r.symbol for r in results if r.eligible} == {"XYZ1", "XYZ2"}


def test_historical_pit_future_rows_are_not_consumed():
    as_of = date(2026, 6, 15)
    df_raw = _generate_ohlcv_df("SYNTH1", 100, end_date_str="2026-06-30")
    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["SYNTH1"], as_of_date=as_of, market_data_map={"SYNTH1": df_raw}
    )
    assert results[0].eligible is True
    assert results[0].pit_safe is True


def test_future_price_cannot_change_candidacy():
    as_of = date(2026, 6, 15)
    df1 = _generate_ohlcv_df("MUTATE_TEST", 60, end_date_str="2026-06-30")
    results1 = CandidateDiscoveryEngine.discover_candidates(
        universe=["MUTATE_TEST"], as_of_date=as_of, market_data_map={"MUTATE_TEST": df1}
    )
    df2 = df1.copy()
    future_mask = df2["timestamp"] > pd.to_datetime(as_of)
    df2.loc[future_mask, "close"] = 999999.0
    df2.loc[future_mask, "high"] = 999999.0
    results2 = CandidateDiscoveryEngine.discover_candidates(
        universe=["MUTATE_TEST"], as_of_date=as_of, market_data_map={"MUTATE_TEST": df2}
    )
    assert results1[0].eligible == results2[0].eligible
    assert results1[0].passed_filters == results2[0].passed_filters


def test_insufficient_history():
    as_of = date(2026, 6, 30)
    df_short = _generate_ohlcv_df("SHORT_HIST", 25)
    result = CandidateDiscoveryEngine.discover_candidates(
        universe=["SHORT_HIST"],
        as_of_date=as_of,
        market_data_map={"SHORT_HIST": df_short},
        config=CandidateDiscoveryConfig(min_history_length=50),
    )[0]
    assert result.eligible is False
    assert "HISTORY_SUFFICIENCY" in result.failed_filters
    assert "INSUFFICIENT_HISTORY" in result.reasons


def test_liquidity():
    as_of = date(2026, 6, 30)
    df_illiquid = _generate_ohlcv_df("ILLIQUID", 60, base_volume=100)
    result = CandidateDiscoveryEngine.discover_candidates(
        universe=["ILLIQUID"],
        as_of_date=as_of,
        market_data_map={"ILLIQUID": df_illiquid},
        config=CandidateDiscoveryConfig(min_average_volume=10000),
    )[0]
    assert result.eligible is False
    assert "LIQUIDITY" in result.failed_filters
    assert "INSUFFICIENT_LIQUIDITY" in result.reasons


def test_data_quality():
    as_of = date(2026, 6, 30)
    df_bad = _generate_ohlcv_df("BAD_GEOM", 60)
    df_bad.loc[30, "high"] = 50.0
    df_bad.loc[30, "low"] = 100.0
    result = CandidateDiscoveryEngine.discover_candidates(
        universe=["BAD_GEOM"], as_of_date=as_of, market_data_map={"BAD_GEOM": df_bad}
    )[0]
    assert result.eligible is False
    assert "DATA_QUALITY" in result.failed_filters
    assert "DATA_QUALITY_INVALID" in result.reasons or "OHLC_INVALID_GEOMETRY" in result.reasons


def test_explanation():
    as_of = date(2026, 6, 30)
    df_penny = _generate_ohlcv_df("PENNY", 60, base_price=5.0)
    result = CandidateDiscoveryEngine.discover_candidates(
        universe=["PENNY"],
        as_of_date=as_of,
        market_data_map={"PENNY": df_penny},
        config=CandidateDiscoveryConfig(min_price=20.0),
    )[0]
    assert result.eligible is False
    assert result.failed_filters
    assert result.reasons
    assert "PRICE_BELOW_MINIMUM" in result.reasons


def test_historical_universe_fail_closed():
    with pytest.raises(HistoricalUniverseUnavailableError):
        HistoricalUniverseProvider.get_universe_for_date(date(2025, 6, 1))


def test_historical_mode_requires_security_metadata():
    as_of = date(2026, 6, 30)
    market_data = {"AAA": _generate_ohlcv_df("AAA", 60, base_price=200.0, base_volume=100000)}
    with pytest.raises(TypeError, match="SymbolMetadata"):
        CandidateDiscoveryEngine.discover_candidates(
            universe=["AAA"],
            as_of_date=as_of,
            market_data_map=market_data,
            mode="HISTORICAL",
        )


def test_historical_mode_uses_listing_and_delisting_dates():
    as_of = date(2026, 6, 30)
    metadata = [
        SymbolMetadata(symbol="ACTIVE", company_name="Active Co", listing_date=date(2020, 1, 1)),
        SymbolMetadata(symbol="FUTURE", company_name="Future Co", listing_date=date(2027, 1, 1)),
        SymbolMetadata(symbol="DELISTED", company_name="Delisted Co", listing_date=date(2020, 1, 1), delisting_date=date(2026, 6, 30)),
    ]
    market_data = {
        "ACTIVE": _generate_ohlcv_df("ACTIVE", 60, base_price=200.0, base_volume=100000),
        "FUTURE": _generate_ohlcv_df("FUTURE", 60, base_price=200.0, base_volume=100000),
        "DELISTED": _generate_ohlcv_df("DELISTED", 60, base_price=200.0, base_volume=100000),
    }
    results = CandidateDiscoveryEngine.discover_candidates(
        universe=metadata,
        as_of_date=as_of,
        market_data_map=market_data,
        mode="HISTORICAL",
    )
    symbols = {r.symbol for r in results}
    assert symbols == {"ACTIVE"}
    assert results[0].eligible is True


def test_live_mode_uses_same_screening_logic():
    as_of = date(2026, 8, 19)
    market_data = {"LIVE1": _generate_ohlcv_df("LIVE1", 60, end_date_str="2026-08-19")}
    results = CandidateDiscoveryEngine.discover_candidates(
        universe=["LIVE1"], as_of_date=as_of, market_data_map=market_data, mode="LIVE"
    )
    assert len(results) == 1
    assert results[0].eligible is True
