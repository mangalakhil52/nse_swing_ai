"""
Unit tests for Stage-1 Quant Screener, Relative Strength, and Market Regime Classifier.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.models import SymbolMetadata
from src.core.types import MarketRegime, TradingStance
from src.quant.indicators import TechnicalIndicators
from src.quant.regime import MarketRegimeClassifier
from src.quant.relative_strength import RelativeStrengthEngine
from src.quant.screener import QuantScreener


def test_relative_strength_engine():
    # Stock outperforming benchmark
    bench_close = pd.Series(np.linspace(100, 110, 60))
    stock_close = pd.Series(np.linspace(100, 135, 60))

    mansfield_rs = RelativeStrengthEngine.calculate_mansfield_rs(stock_close, bench_close, period=50)
    assert mansfield_rs.iloc[-1] > 0.0

    alphas = RelativeStrengthEngine.calculate_multi_period_alpha(stock_close, bench_close)
    assert alphas["alpha_20d"] > 0.0

    ranks = RelativeStrengthEngine.calculate_universe_percentile_ranks({"TRENT": 15.0, "RELIANCE": 5.0, "WEAK": -8.0})
    assert ranks["TRENT"] > ranks["RELIANCE"] > ranks["WEAK"]


def test_market_regime_classification():
    # Bullish Nifty series
    nifty_close = np.linspace(20000, 25000, 100)
    high = nifty_close * 1.01
    low = nifty_close * 0.99
    open_p = nifty_close - 10.0
    vol = np.full(100, 5000000)

    nifty_df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": nifty_close, "volume": vol})
    res = MarketRegimeClassifier.classify_regime(
        nifty_df, advance_decline_ratio=1.8, pct_above_50_sma=75.0, india_vix=13.5
    )

    assert res.regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]
    assert res.trading_stance in [TradingStance.AGGRESSIVE, TradingStance.NORMAL]
    assert res.allow_long_swing_trades is True


def test_quant_screener_filtering():
    screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)

    # 1. Good candidate (strong uptrend, tight range near 52W high, liquid)
    good_close = np.linspace(100, 160, 100)
    good_df = pd.DataFrame({
        "open": good_close - 1.0,
        "high": good_close + 2.0,
        "low": good_close - 2.0,
        "close": good_close,
        "volume": np.full(100, 1000000),
        "turnover_crores": np.full(100, 15.0),
    })
    sec_good = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", is_fno_eligible=True)

    cand = screener.screen_single_stock(sec_good, good_df)
    assert cand is not None
    assert cand.symbol == "TRENT"
    assert cand.adtv_crores >= 5.0

    # 2. Illiquid candidate (turnover 0.5 Cr < 5.0 Cr)
    illiquid_df = good_df.copy()
    illiquid_df["turnover_crores"] = 0.5
    sec_illiquid = SymbolMetadata(symbol="PENNY", company_name="Penny Co")
    cand_illiquid = screener.screen_single_stock(sec_illiquid, illiquid_df)
    assert cand_illiquid is None

    # 3. Downtrend candidate (Close < 20 EMA)
    down_close = np.linspace(200, 100, 100)
    down_df = pd.DataFrame({
        "open": down_close + 1.0,
        "high": down_close + 2.0,
        "low": down_close - 2.0,
        "close": down_close,
        "volume": np.full(100, 1000000),
        "turnover_crores": np.full(100, 10.0),
    })
    sec_down = SymbolMetadata(symbol="WEAK", company_name="Weak Trend Ltd")
    cand_down = screener.screen_single_stock(sec_down, down_df)
    assert cand_down is None
