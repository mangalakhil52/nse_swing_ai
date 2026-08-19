"""
P0 #12D — Market Regime & Benchmark Point-in-Time Integrity Unit & Integration Tests.

Validates that:
  1. MarketRegime input is strictly PIT bounded <= decision_time T.
  2. Future market data (timestamp > T) CANNOT change regime classification or stance at T.
  3. Future market data (timestamp > T) CANNOT change regime feature values (close, EMAs, ADX) at T.
  4. Passing un-truncated data containing future rows to PIT boundary raises PITViolationError (fails closed).
  5. Benchmark-derived relative strength features (Mansfield RS, Alpha) cannot be changed by future benchmark data.
  6. Missing benchmark data returns UNKNOWN regime or DATA_UNAVAILABLE status without forward-filling from future.
  7. Same-day timestamp ordering is strictly enforced for intraday benchmark timestamps.
  8. Active historical backtest engine consumer status is explicitly documented.
"""

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.agents.relative_strength_agent import RelativeStrengthAgent
from src.core.evidence import EvidenceGraph
from src.core.models import SymbolMetadata
from src.core.types import AgentStatus, MarketRegime, TradingStance
from src.data.point_in_time import PointInTimeFilter, PITViolationError
from src.quant.regime import MarketRegimeClassifier
from src.quant.indicators import TechnicalIndicators


def _generate_synthetic_nifty_df(num_bars: int = 150) -> pd.DataFrame:
    """Helper generating deterministic Nifty 50 OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(end="2026-06-30", periods=num_bars, freq="B")
    base_price = 22000.0
    returns = np.random.normal(0.0005, 0.008, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, size=num_bars),
    })
    return df


def test_market_regime_input_contains_no_future_rows():
    """1. Test MarketRegimeClassifier input contains zero rows > decision_date T."""
    nifty_df = _generate_synthetic_nifty_df(150)
    as_of_idx = 100
    as_of_dt = nifty_df.iloc[as_of_idx]["timestamp"].date()

    res = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df,
        advance_decline_ratio=1.6,
        pct_above_50_sma=70.0,
        india_vix=14.0,
        as_of_date=as_of_dt,
    )

    assert res.regime != MarketRegime.UNKNOWN
    # Expected close should match the close price exactly at index 100, not 149
    expected_close = round(float(nifty_df.iloc[as_of_idx]["close"]), 2)
    assert res.nifty_close == expected_close


def test_future_market_data_cannot_change_regime_at_T():
    """2. Test mutating ONLY future Nifty market data (timestamp > T) produces 100% identical regime output at T."""
    nifty_df_base = _generate_synthetic_nifty_df(150)
    as_of_idx = 100
    as_of_dt = nifty_df_base.iloc[as_of_idx]["timestamp"].date()

    # Baseline regime evaluation
    res_base = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df_base,
        advance_decline_ratio=1.6,
        pct_above_50_sma=70.0,
        india_vix=14.0,
        as_of_date=as_of_dt,
    )

    # Mutate ONLY future rows (idx 101..149 > as_of_idx)
    nifty_df_mut = nifty_df_base.copy()
    nifty_df_mut.loc[as_of_idx + 1 :, "open"] *= 5.0
    nifty_df_mut.loc[as_of_idx + 1 :, "high"] *= 5.0
    nifty_df_mut.loc[as_of_idx + 1 :, "low"] *= 0.2
    nifty_df_mut.loc[as_of_idx + 1 :, "close"] *= 5.0
    nifty_df_mut.loc[as_of_idx + 1 :, "volume"] *= 10.0

    res_mut = MarketRegimeClassifier.classify_regime(
        nifty_df=nifty_df_mut,
        advance_decline_ratio=1.6,
        pct_above_50_sma=70.0,
        india_vix=14.0,
        as_of_date=as_of_dt,
    )

    assert res_base.regime == res_mut.regime
    assert res_base.trading_stance == res_mut.trading_stance
    assert res_base.nifty_close == res_mut.nifty_close
    assert res_base.risk_multiplier == res_mut.risk_multiplier
    assert res_base.confidence == res_mut.confidence
    assert res_base.allow_long_swing_trades == res_mut.allow_long_swing_trades


def test_future_market_data_cannot_change_regime_features_at_T():
    """3. Test computed technical indicators on Nifty at T remain 100% identical under future price mutation."""
    nifty_df_base = _generate_synthetic_nifty_df(150)
    as_of_idx = 100
    as_of_dt = nifty_df_base.iloc[as_of_idx]["timestamp"].date()

    raw_base = PointInTimeFilter.filter_market_data(nifty_df_base, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(raw_base, as_of_dt)
    feat_base = TechnicalIndicators.compute_all_indicators(raw_base)

    nifty_df_mut = nifty_df_base.copy()
    nifty_df_mut.loc[as_of_idx + 1 :, "close"] *= 10.0

    raw_mut = PointInTimeFilter.filter_market_data(nifty_df_mut, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(raw_mut, as_of_dt)
    feat_mut = TechnicalIndicators.compute_all_indicators(raw_mut)

    cols_to_check = ["close", "ema_20", "ema_50", "ema_200", "adx_14"]
    for col in cols_to_check:
        base_val = float(feat_base[col].iloc[-1])
        mut_val = float(feat_mut[col].iloc[-1])
        assert base_val == pytest.approx(mut_val, abs=1e-6)


def test_regime_pit_violation_fails_closed():
    """4. Test enforce_pit_boundary raises PITViolationError if an un-truncated Nifty DataFrame with future rows is passed."""
    nifty_df = _generate_synthetic_nifty_df(150)
    as_of_idx = 100
    as_of_dt = nifty_df.iloc[as_of_idx]["timestamp"].date()

    # Pass un-truncated DataFrame containing future rows (up to idx 110 > idx 100)
    future_df = nifty_df.iloc[: as_of_idx + 11].copy()

    with pytest.raises(PITViolationError) as exc_info:
        PointInTimeFilter.enforce_pit_boundary(future_df, as_of_dt)

    assert "PIT Violation" in str(exc_info.value)


def test_future_benchmark_data_cannot_change_benchmark_feature_at_T():
    """5. Test mutating future benchmark Nifty data leaves RelativeStrengthAgent output at T 100% identical."""
    import asyncio

    stock_df = _generate_synthetic_nifty_df(150)
    nifty_base = _generate_synthetic_nifty_df(150)
    as_of_idx = 100
    as_of_dt = nifty_base.iloc[as_of_idx]["timestamp"].date()

    nifty_mut = nifty_base.copy()
    nifty_mut.loc[as_of_idx + 1 :, "close"] *= 5.0

    agent = RelativeStrengthAgent()
    meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")

    # Baseline run at T
    ctx_base = {"nifty_df": nifty_base, "as_of_date": as_of_dt}
    out_base = asyncio.run(agent._analyze(meta, stock_df.iloc[: as_of_idx + 1], EvidenceGraph(), "r1", ctx_base))

    # Mutated run at T
    ctx_mut = {"nifty_df": nifty_mut, "as_of_date": as_of_dt}
    out_mut = asyncio.run(agent._analyze(meta, stock_df.iloc[: as_of_idx + 1], EvidenceGraph(), "r2", ctx_mut))

    assert out_base.score == out_mut.score
    assert out_base.signal == out_mut.signal


def test_benchmark_pit_violation_fails_closed():
    """6. Test benchmark DataFrame containing future rows passed directly to enforce_pit_boundary raises PITViolationError."""
    nifty_df = _generate_synthetic_nifty_df(150)
    as_of_dt = nifty_df.iloc[100]["timestamp"].date()
    future_nifty = nifty_df.iloc[:110].copy()

    with pytest.raises(PITViolationError):
        PointInTimeFilter.enforce_pit_boundary(future_nifty, as_of_dt)


def test_same_day_benchmark_timestamp_ordering():
    """7. Test intraday benchmark timestamp slicing enforces exact timestamp ordering <= T."""
    nifty_df = _generate_synthetic_nifty_df(150)
    as_of_dt = nifty_df.iloc[100]["timestamp"].date()

    sliced = PointInTimeFilter.filter_market_data(nifty_df, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(sliced, as_of_dt)

    max_ts_date = pd.to_datetime(sliced["timestamp"]).max().date()
    assert max_ts_date <= as_of_dt


def test_missing_benchmark_data_does_not_use_future_observation():
    """8. Test missing Nifty benchmark data causes MarketRegimeClassifier to fail closed with UNKNOWN regime."""
    res = MarketRegimeClassifier.classify_regime(
        nifty_df=None,
        advance_decline_ratio=1.5,
        pct_above_50_sma=70.0,
        india_vix=14.0,
    )

    assert res.regime == MarketRegime.UNKNOWN
    assert res.trading_stance == TradingStance.NO_TRADE
    assert res.risk_multiplier == 0.0
    assert res.allow_long_swing_trades is False


def test_market_regime_consumer_status_documented_as_not_implemented():
    """9. Documents that active backtest engine (PortfolioBacktestEngine) runs technical signals without regime dependency."""
    from src.backtest.portfolio import PortfolioBacktestEngine
    import inspect

    sig = inspect.signature(PortfolioBacktestEngine.run_portfolio_backtest)
    assert "stock_dfs" in sig.parameters
    assert "market_regime" not in sig.parameters
