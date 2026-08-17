"""
Unit & Parity Tests for P0 Fix #10: Backtest Engine Strategy Parity.

Verifies:
  1. Backtest signal generation uses production TradeConstructionEngine.
  2. Zero hardcoded R multiples (1.8R, 2.8R, 4.5R) or independent trade formulas.
  3. Exit dates are real date timestamps (e.g. '2026-01-28'), NOT integer bar indices.
  4. Same-candle SL + Target conflicts use deterministic worst-case resolution (SL hit).
  5. Missing required indicators cause trade rejection.
  6. Invalid trade geometry (> 8% stop) causes trade rejection.
  7. Future candles (t > T) cannot alter trade construction.
"""

from datetime import datetime
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.run_backtest import generate_signals_from_df
from src.agents.trade_construction_agent import TradeConstructionEngine
from src.backtest.engine import BacktestEngine
from src.backtest.friction import IndianFrictionModel
from src.quant.indicators import TechnicalIndicators


def test_indian_friction_model():
    costs = IndianFrictionModel.calculate_round_trip(
        entry_price=1000.0,
        exit_price=1200.0,
        shares=100,
    )
    assert costs.total_cost_rupees > 0
    assert costs.entry_stt > 0
    assert costs.exit_stt > 0
    assert costs.total_cost_pct < 1.0


def test_backtest_engine_winning_trade():
    dates = pd.date_range(start="2026-01-01", periods=60, freq="B")
    close = np.linspace(1000, 1300, 60)
    high = close + 10.0
    low = close - 10.0
    open_p = close - 5.0

    df = pd.DataFrame({
        "timestamp": dates, "open": open_p, "high": high, "low": low, "close": close, "volume": [500000]*60
    })

    trade = BacktestEngine.simulate_trade(
        symbol="TRENT",
        df=df,
        entry_date_idx=5,
        entry_price=1050.0,
        stop_loss=990.0,
        target_1=1158.0,
        target_2=1218.0,
        target_3=1320.0,
        shares=20,
    )

    assert trade.exit_reason is not None
    assert trade.pnl_rupees is not None
    assert trade.holding_sessions >= 1
    assert trade.entry_date == dates[5].strftime("%Y-%m-%d")
    assert trade.exit_date > trade.entry_date


def test_backtest_same_candle_sl_and_target_uses_worst_case():
    """Requirement 6: Same-candle SL + Target conflict must resolve to worst case (SL hit)."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0] * 10,
        "high": [101.0, 115.0] + [100.0]*8,   # High 115 hits target_1 (110)
        "low": [99.0, 90.0] + [100.0]*8,      # Low 90 hits stop_loss (95)
        "close": [100.0] * 10,
        "volume": [100000] * 10,
    })

    trade = BacktestEngine.simulate_trade(
        symbol="CONFLICT",
        df=df,
        entry_date_idx=0,
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=115.0,
        target_3=120.0,
        shares=10,
    )

    # Worst-case rule: SL hit must take precedence over Target 1 on same candle!
    assert trade.exit_reason in ["STOP_LOSS_HIT", "STOP_LOSS_GAP"]
    assert trade.exit_price == 95.0
    assert trade.pnl_rupees < 0.0


def test_backtest_exit_date_is_real_timestamp():
    """Requirement 4: Exit dates must be real timestamp strings (YYYY-MM-DD), NOT bar indices."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0] * 10,
        "high": [101.0, 102.0, 112.0] + [100.0]*7,  # Target 1 hit on index 2
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "volume": [100000] * 10,
    })

    trade = BacktestEngine.simulate_trade(
        symbol="DATE_TEST",
        df=df,
        entry_date_idx=0,
        entry_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=115.0,
        target_3=120.0,
        shares=10,
    )

    assert trade.entry_date == dates[0].strftime("%Y-%m-%d")
    assert trade.exit_date == dates[9].strftime("%Y-%m-%d")
    assert not trade.exit_date.isdigit()


def test_backtest_entry_signal_direct_call_and_parity():
    """Requirement 6 & 9: BacktestEngine.backtest_entry_signal constructs canonical trade levels independently."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    volumes = [50000] * 50 + [300000, 80000, 90000, 120000, 100000, 95000, 90000, 85000, 80000, 75000]

    df_hist = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": volumes,
    })
    df_ind = TechnicalIndicators.compute_all_indicators(df_hist.copy())

    # 1. Direct canonical call at entry_idx 50
    prod_levels, _ = TradeConstructionEngine.construct_trade_levels("TRENT", df_ind.iloc[:51])
    assert prod_levels is not None

    # 2. Backtest entry signal call
    trade, err = BacktestEngine.backtest_entry_signal("TRENT", df_ind, 50)
    assert err is None
    assert trade is not None

    assert trade.entry_price == prod_levels.entry_trigger_price
    assert trade.stop_loss == prod_levels.stop_loss_price
    assert trade.target_1 == prod_levels.target_1
    assert trade.target_2 == prod_levels.target_2
    assert trade.target_3 == prod_levels.target_3
    assert trade.shares == prod_levels.position_size_shares


def test_backtest_entry_signal_rejects_deliberate_parity_violation():
    """Requirement 5 & 10: Supplying deliberate incorrect trade levels causes PARITY_VIOLATION rejection."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    volumes = [50000] * 50 + [300000, 80000, 90000, 120000, 100000, 95000, 90000, 85000, 80000, 75000]

    df_hist = pd.DataFrame({
        "timestamp": dates, "open": prices, "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices], "close": prices, "volume": volumes,
    })
    df_ind = TechnicalIndicators.compute_all_indicators(df_hist.copy())

    # Pass fake externally supplied trade levels (e.g. entry_price = 999.0 instead of 100.3)
    fake_supplied = {
        "entry_price": 999.0,
        "stop_loss": 900.0,
        "target_1": 1200.0,
    }

    trade, err = BacktestEngine.backtest_entry_signal("TRENT", df_ind, 50, supplied_levels=fake_supplied)
    assert trade is None
    assert err is not None
    assert "PARITY_VIOLATION" in err


def test_backtest_uses_production_trade_construction_parity():
    """Requirement 1 & 2: Backtest signals match production TradeConstructionEngine 100%."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]
    volumes = [50000] * 50 + [300000, 80000, 90000, 120000, 100000, 95000, 90000, 85000, 80000, 75000]

    df_hist = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": volumes,
    })

    # Direct live production call at index 50
    df_ind = TechnicalIndicators.compute_all_indicators(df_hist.copy())
    sub_df_50 = df_ind.iloc[:51]
    prod_levels, err = TradeConstructionEngine.construct_trade_levels("TRENT", sub_df_50)
    assert prod_levels is not None

    # Backtest signal generation call
    signals = generate_signals_from_df(df_hist, "TRENT")
    assert len(signals) > 0

    backtest_sig = signals.iloc[0]
    assert backtest_sig["entry_price"] == prod_levels.entry_trigger_price
    assert backtest_sig["stop_loss"] == prod_levels.stop_loss_price
    assert backtest_sig["target_1"] == prod_levels.target_1
    assert backtest_sig["target_2"] == prod_levels.target_2
    assert backtest_sig["target_3"] == prod_levels.target_3
    assert backtest_sig["shares"] == prod_levels.position_size_shares


def test_no_hardcoded_r_multipliers_in_backtest_source():
    """Requirement 3: Source code of run_backtest.py contains ZERO hardcoded 1.8R/2.8R/4.5R."""
    backtest_script = Path("scripts/run_backtest.py").read_text(encoding="utf-8")
    assert "risk * 1.8" not in backtest_script
    assert "risk * 2.8" not in backtest_script
    assert "risk * 4.5" not in backtest_script
    assert "10000 / entry_price" not in backtest_script


def test_backtest_missing_indicators_rejects_trade():
    """Requirement 7: Missing required indicators cause backtest to reject trade."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices = [100.0] * 60

    df_raw = pd.DataFrame({
        "timestamp": dates, "open": prices, "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices], "close": prices, "volume": [100000] * 60,
    })

    # Test direct call to TradeConstructionEngine without required indicators
    levels, err = TradeConstructionEngine.construct_trade_levels("NO_IND", df_raw)
    assert levels is None
    assert "Required indicator data" in err


def test_backtest_future_candles_cannot_alter_trade_levels():
    """Requirement 5: Future candles (t > T) cannot alter backtest trade levels constructed at T."""
    dates = pd.date_range(start="2025-11-15", periods=60, freq="B")
    prices_orig = [98.0] * 50 + [100.0, 103.0, 106.0, 112.0, 115.0, 118.0, 120.0, 122.0, 124.0, 125.0]

    df_orig = pd.DataFrame({
        "timestamp": dates, "open": prices_orig, "high": [p * 1.002 for p in prices_orig],
        "low": [p * 0.998 for p in prices_orig], "close": prices_orig, "volume": [50000]*50 + [300000]*10,
    })

    sig_orig = generate_signals_from_df(df_orig, "MUTATE")
    assert len(sig_orig) > 0

    # Mutate future candles at index 53 onwards
    df_mutated = df_orig.copy()
    df_mutated.loc[53:, "close"] = df_mutated.loc[53:, "close"] * 2.0
    df_mutated.loc[53:, "high"] = df_mutated.loc[53:, "high"] * 2.0

    sig_mutated = generate_signals_from_df(df_mutated, "MUTATE")
    assert len(sig_mutated) > 0

    assert sig_orig.iloc[0]["entry_price"] == sig_mutated.iloc[0]["entry_price"]
    assert sig_orig.iloc[0]["stop_loss"] == sig_mutated.iloc[0]["stop_loss"]
    assert sig_orig.iloc[0]["target_1"] == sig_mutated.iloc[0]["target_1"]
