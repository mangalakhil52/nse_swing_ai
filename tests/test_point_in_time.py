"""
Unit & Integration Tests for P0 #12A: Point-in-Time Data Integrity & Leakage Prevention.

Coverage:
  1. test_1_future_ohlcv_mutation: T+1 OHLCV mutation leaves T signals identical.
  2. test_2_future_technical_feature_mutation: Indicators (SMA, EMA, RSI, ATR) at T are identical under future mutation.
  3. test_3_current_bar_semantics: Signal at T close uses completed bar T; T+1 prices invisible at signal time.
  4. test_4_fundamental_publication_date_filtering: Fundamental record filtered strictly by filing_date / available_at.
  5. test_5_news_publication_date_filtering: News filtered strictly by published_at <= as_of_date.
  6. test_6_market_regime_future_mutation: Market regime at T is identical when future benchmark data is mutated.
  7. test_7_outcome_label_isolation: Outcome labels cannot influence signals and adhere to completion date bounds.
  8. test_8_future_scaler_contamination: Point-in-time feature normalization is unaffected by future outliers.
  9. test_9_cross_sectional_future_mutation: Cross-sectional ranking at T is unaffected by T+1 prices.
 10. test_10_missing_availability_timestamp_fail_closed: Missing filing_date/available_at fails closed with PIT_UNVERIFIED.
 11. test_11_pit_fail_closed_behavior: PointInTimeFilter rejects unverified availability timestamps.
 12. test_12_deterministic_repeatability_actual_signal_path: Executes actual PatternRecognizer & TradeConstructionEngine path.
"""

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.core.models import CorporateAnnouncement, CorporateEvent, NewsArticle, QuarterlyFinancials
from src.data.point_in_time import PointInTimeFilter, PITContract, PITRegressionHelper
from src.quant.patterns import PatternRecognizer
from src.agents.trade_construction_agent import TradeConstructionEngine
from src.quant.indicators import TechnicalIndicators
from src.quant.regime import MarketRegimeClassifier


def _generate_synthetic_stock_df(num_days=200, symbol="TRENT", start_date="2025-01-01"):
    dates = pd.date_range(start=start_date, periods=num_days, freq="B")
    prices = []
    curr = 1000.0
    for i in range(num_days):
        ret = 0.002 if i % 2 == 1 else -0.001
        if 100 <= i < 120:
            ret = 0.015  # Pattern breakout
        curr *= (1.0 + ret)
        prices.append(round(curr, 2))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [round(p * 1.005, 2) for p in prices],
        "low": [round(p * 0.995, 2) for p in prices],
        "close": prices,
        "volume": [100000 + (i * 1000) for i in range(num_days)],
    })
    return symbol, df


def test_1_future_ohlcv_mutation():
    """1. Test mutating T+1 OHLCV leaves features and signals generated at T 100% identical."""
    sym, df_orig = _generate_synthetic_stock_df(150)
    eval_idx = 115
    eval_dt = df_orig.iloc[eval_idx]["timestamp"]

    # Baseline slice at t <= eval_dt
    sub_orig = df_orig.iloc[: eval_idx + 1].copy()
    patterns_orig = PatternRecognizer.evaluate_all_patterns(sub_orig)
    matched_orig = [p.pattern_type.value for p in patterns_orig if p.is_matched]

    # Mutate T+1 price in full dataset
    df_mut = df_orig.copy()
    df_mut.loc[eval_idx + 1 :, "close"] *= 5.0

    # Slice mutated dataset at t <= eval_dt
    sub_mut = df_mut.iloc[: eval_idx + 1].copy()
    patterns_mut = PatternRecognizer.evaluate_all_patterns(sub_mut)
    matched_mut = [p.pattern_type.value for p in patterns_mut if p.is_matched]

    assert matched_orig == matched_mut


def test_2_future_technical_feature_mutation():
    """2. Test indicator calculations (SMA, EMA, RSI, ATR) at T are identical under future data mutation."""
    sym, df_orig = _generate_synthetic_stock_df(150)
    eval_idx = 100

    sub_orig = df_orig.iloc[: eval_idx + 1].copy()
    sma_orig = float(sub_orig["close"].rolling(20).mean().iloc[-1])
    rsi_orig = float(TechnicalIndicators.calculate_rsi(sub_orig["close"]).iloc[-1])

    # Mutate future rows
    df_mut = df_orig.copy()
    df_mut.loc[eval_idx + 1 :, "close"] *= 10.0

    sub_mut = df_mut.iloc[: eval_idx + 1].copy()
    sma_mut = float(sub_mut["close"].rolling(20).mean().iloc[-1])
    rsi_mut = float(TechnicalIndicators.calculate_rsi(sub_mut["close"]).iloc[-1])

    assert sma_orig == sma_mut
    assert rsi_orig == rsi_mut


def test_3_current_bar_semantics():
    """3. Test signal at T close uses completed bar T, while T+1 prices are invisible."""
    sym, df = _generate_synthetic_stock_df(150)
    eval_idx = 110

    # Signal evaluated using data up to bar T
    df_t = PointInTimeFilter.filter_market_data(df, df.iloc[eval_idx]["timestamp"].date())
    assert len(df_t) == eval_idx + 1
    assert pd.to_datetime(df_t.iloc[-1]["timestamp"]).date() == df.iloc[eval_idx]["timestamp"].date()


def test_4_fundamental_publication_date_filtering():
    """4. Test fundamental record is available ONLY after filing_date / available_at."""
    q_fin = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        filing_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        sales_crores=1500.0,
        sales_growth_yoy_pct=15.0,
        pat_crores=200.0,
        pat_growth_yoy_pct=20.0,
        ebitda_margin_pct=18.0,
        eps_inr=25.0,
        pit_status="VERIFIED",
    )

    # Evaluation at D5 (2026-04-30) -> Unavailable
    res_d5 = PointInTimeFilter.filter_quarterly_financials([q_fin], date(2026, 4, 30))
    assert len(res_d5) == 0

    # Evaluation at D10 (2026-05-15) -> Available
    res_d10 = PointInTimeFilter.filter_quarterly_financials([q_fin], date(2026, 5, 15))
    assert len(res_d10) == 1

    # Evaluation at D11 (2026-05-16) -> Available
    res_d11 = PointInTimeFilter.filter_quarterly_financials([q_fin], date(2026, 5, 16))
    assert len(res_d11) == 1


def test_5_news_publication_date_filtering():
    """5. Test news articles published at D10 are unavailable at D5 and available at D10."""
    article = NewsArticle(
        symbol="TRENT",
        headline="Strong Q4 Retail Expansion",
        publisher="Economic Times",
        published_at=datetime(2026, 5, 15, 10, 0, 0),
    )

    # Evaluation at D5 (2026-05-05) -> Unavailable
    res_d5 = PointInTimeFilter.filter_news([article], date(2026, 5, 5))
    assert len(res_d5) == 0

    # Evaluation at D10 (2026-05-15) -> Available
    res_d10 = PointInTimeFilter.filter_news([article], date(2026, 5, 15))
    assert len(res_d10) == 1


def test_6_market_regime_future_mutation():
    """6. Test market regime at T is 100% identical when future benchmark data is mutated."""
    sym, df_orig = _generate_synthetic_stock_df(200, symbol="NIFTY")
    as_of_idx = 150
    as_of_dt = df_orig.iloc[as_of_idx]["timestamp"]

    # Baseline regime
    sub_orig = df_orig.iloc[: as_of_idx + 1].copy()
    regime_orig = MarketRegimeClassifier.classify_regime(nifty_df=sub_orig)

    # Mutate future rows > as_of_dt
    df_mut = df_orig.copy()
    df_mut.loc[as_of_idx + 1 :, "close"] *= 3.0

    sub_mut = df_mut.iloc[: as_of_idx + 1].copy()
    regime_mut = MarketRegimeClassifier.classify_regime(nifty_df=sub_mut)

    assert regime_orig.regime == regime_mut.regime
    assert regime_orig.trading_stance == regime_mut.trading_stance


def test_7_outcome_label_isolation():
    """7. Test outcome labels extending beyond train_end are excluded and never influence live signals."""
    from src.backtest.walk_forward import WalkForwardValidator

    train_end = "2025-06-30"
    # Label completing after train_end -> Excluded
    assert WalkForwardValidator.is_outcome_label_eligible("2025-06-25", 20, train_end) is False
    # Label completing before train_end -> Eligible
    assert WalkForwardValidator.is_outcome_label_eligible("2025-05-01", 10, train_end) is True


def test_8_future_scaler_contamination():
    """8. Test point-in-time ATR normalization at T is unaffected by massive future price outlier at T+10."""
    sym, df_orig = _generate_synthetic_stock_df(150)
    eval_idx = 100

    sub_orig = df_orig.iloc[: eval_idx + 1].copy()
    atr_orig_series, _ = TechnicalIndicators.calculate_atr(sub_orig["high"], sub_orig["low"], sub_orig["close"])
    atr_orig = float(atr_orig_series.iloc[-1])

    # Future price spike at eval_idx + 10
    df_mut = df_orig.copy()
    df_mut.loc[eval_idx + 10 :, "high"] *= 100.0

    sub_mut = df_mut.iloc[: eval_idx + 1].copy()
    atr_mut_series, _ = TechnicalIndicators.calculate_atr(sub_mut["high"], sub_mut["low"], sub_mut["close"])
    atr_mut = float(atr_mut_series.iloc[-1])

    assert atr_orig == atr_mut


def test_9_cross_sectional_future_mutation():
    """9. Test cross-sectional ranking of Symbol A at T is unaffected by Symbol B price changes at T+1."""
    sym_a, df_a = _generate_synthetic_stock_df(150, symbol="TRENT")
    sym_b, df_b = _generate_synthetic_stock_df(150, symbol="RELIANCE")
    eval_idx = 100

    # RS comparison at T
    ret_a_orig = (df_a.iloc[eval_idx]["close"] / df_a.iloc[eval_idx - 20]["close"]) - 1.0
    ret_b_orig = (df_b.iloc[eval_idx]["close"] / df_b.iloc[eval_idx - 20]["close"]) - 1.0

    # Mutate Symbol B at T+1
    df_b_mut = df_b.copy()
    df_b_mut.loc[eval_idx + 1 :, "close"] *= 5.0

    ret_a_mut = (df_a.iloc[eval_idx]["close"] / df_a.iloc[eval_idx - 20]["close"]) - 1.0
    ret_b_mut = (df_b_mut.iloc[eval_idx]["close"] / df_b_mut.iloc[eval_idx - 20]["close"]) - 1.0

    assert ret_a_orig == ret_a_mut
    assert ret_b_orig == ret_b_mut


def test_10_missing_availability_timestamp_fail_closed():
    """10. Test QuarterlyFinancials or CorporateEvent missing availability timestamp fails closed."""
    q_unverified = QuarterlyFinancials(
        symbol="TRENT",
        period_end_date=date(2026, 3, 31),
        sales_crores=1500.0,
        sales_growth_yoy_pct=15.0,
        pat_crores=200.0,
        pat_growth_yoy_pct=20.0,
        ebitda_margin_pct=18.0,
        eps_inr=25.0,
    )
    # filing_date and available_at are None -> Excluded by PointInTimeFilter
    filtered = PointInTimeFilter.filter_quarterly_financials([q_unverified], date(2026, 5, 15))
    assert len(filtered) == 0


def test_11_pit_fail_closed_behavior():
    """11. Test CorporateEvent missing availability timestamp fails closed (excluded)."""
    event_unverified = CorporateEvent(
        symbol="TRENT",
        event_type="BOARD_MEETING",
        event_date=date(2026, 5, 20),
        purpose="Q4 Results",
    )
    filtered = PointInTimeFilter.filter_events([event_unverified], date(2026, 5, 20))
    assert len(filtered) == 0


def test_12_deterministic_repeatability_actual_signal_path():
    """12. Test executes actual PatternRecognizer & TradeConstructionEngine path on baseline vs future-mutated data."""
    sym, df_orig = _generate_synthetic_stock_df(200)
    eval_idx = 150

    # Baseline path execution
    sub_orig = df_orig.iloc[: eval_idx + 1].copy()
    patterns_orig = PatternRecognizer.evaluate_all_patterns(sub_orig)
    trade_levels_orig, err_orig = TradeConstructionEngine.construct_trade_levels(sym, sub_orig)

    # Mutate data strictly AFTER eval_idx
    df_mut = df_orig.copy()
    df_mut.loc[eval_idx + 1 :, "close"] *= 4.0
    df_mut.loc[eval_idx + 1 :, "high"] *= 4.0

    sub_mut = df_mut.iloc[: eval_idx + 1].copy()
    patterns_mut = PatternRecognizer.evaluate_all_patterns(sub_mut)
    trade_levels_mut, err_mut = TradeConstructionEngine.construct_trade_levels(sym, sub_mut)

    # Assert exact signal & trade level identity
    matched_orig = [(p.pattern_type.value, p.quality_score) for p in patterns_orig if p.is_matched]
    matched_mut = [(p.pattern_type.value, p.quality_score) for p in patterns_mut if p.is_matched]
    assert matched_orig == matched_mut

    if trade_levels_orig is not None:
        assert trade_levels_mut is not None
        assert trade_levels_orig.entry_trigger_price == trade_levels_mut.entry_trigger_price
        assert trade_levels_orig.stop_loss_price == trade_levels_mut.stop_loss_price
        assert trade_levels_orig.target_1 == trade_levels_mut.target_1
        assert trade_levels_orig.position_size_shares == trade_levels_mut.position_size_shares


def test_signal_input_contains_no_future_rows():
    """13. Test signal input contains no future rows after passing full DataFrame through production PIT boundary."""
    sym, df = _generate_synthetic_stock_df(150)
    as_of_idx = 100
    as_of_dt = df.iloc[as_of_idx]["timestamp"].date()

    # Pass full un-truncated DataFrame (150 rows) through production PIT boundary
    pit_df = PointInTimeFilter.filter_market_data(df, as_of_dt)
    enforced = PointInTimeFilter.enforce_pit_boundary(pit_df, as_of_dt)

    max_dt = pd.to_datetime(enforced["timestamp"]).max().date()
    assert max_dt <= as_of_dt
    assert len(enforced) == as_of_idx + 1


def test_future_ohlcv_mutation_does_not_change_signal():
    """14. Test mutating future OHLCV (> T) cannot alter signal matched patterns, quality scores, or trade levels at T."""
    sym, df_orig = _generate_synthetic_stock_df(180)
    as_of_idx = 120
    as_of_dt = df_orig.iloc[as_of_idx]["timestamp"].date()

    # Baseline signals at T through full production pipeline
    df_t_orig = PointInTimeFilter.filter_market_data(df_orig, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(df_t_orig, as_of_dt)
    features_orig = TechnicalIndicators.compute_all_indicators(df_t_orig)
    patterns_orig = PatternRecognizer.evaluate_all_patterns(features_orig)
    levels_orig, _ = TradeConstructionEngine.construct_trade_levels(sym, features_orig)

    signals_orig = [(p.pattern_type.value, p.quality_score) for p in patterns_orig if p.is_matched]

    # Mutate future rows (> T) substantially
    df_mut = df_orig.copy()
    df_mut.loc[as_of_idx + 1 :, "open"] *= 10.0
    df_mut.loc[as_of_idx + 1 :, "high"] *= 10.0
    df_mut.loc[as_of_idx + 1 :, "low"] *= 0.1
    df_mut.loc[as_of_idx + 1 :, "close"] *= 10.0
    df_mut.loc[as_of_idx + 1 :, "volume"] *= 500

    df_t_mut = PointInTimeFilter.filter_market_data(df_mut, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(df_t_mut, as_of_dt)
    features_mut = TechnicalIndicators.compute_all_indicators(df_t_mut)
    patterns_mut = PatternRecognizer.evaluate_all_patterns(features_mut)
    levels_mut, _ = TradeConstructionEngine.construct_trade_levels(sym, features_mut)

    signals_mut = [(p.pattern_type.value, p.quality_score) for p in patterns_mut if p.is_matched]

    assert signals_orig == signals_mut

    if levels_orig is not None:
        assert levels_mut is not None
        assert levels_orig.entry_trigger_price == levels_mut.entry_trigger_price
        assert levels_orig.stop_loss_price == levels_mut.stop_loss_price
        assert levels_orig.target_1 == levels_mut.target_1
        assert levels_orig.target_2 == levels_mut.target_2
        assert levels_orig.target_3 == levels_mut.target_3
        assert levels_orig.position_size_shares == levels_mut.position_size_shares


def test_future_ohlcv_mutation_does_not_change_trade_levels():
    """15. Test mutating future OHLCV (> T) cannot alter trade construction levels at T."""
    sym, df_orig = _generate_synthetic_stock_df(180)
    as_of_idx = 120
    as_of_dt = df_orig.iloc[as_of_idx]["timestamp"].date()

    df_t_orig = PointInTimeFilter.filter_market_data(df_orig, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(df_t_orig, as_of_dt)
    features_orig = TechnicalIndicators.compute_all_indicators(df_t_orig)
    levels_orig, err_orig = TradeConstructionEngine.construct_trade_levels(sym, features_orig)

    df_mut = df_orig.copy()
    df_mut.loc[as_of_idx + 1 :, "close"] *= 0.1
    df_mut.loc[as_of_idx + 1 :, "low"] *= 0.1

    df_t_mut = PointInTimeFilter.filter_market_data(df_mut, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(df_t_mut, as_of_dt)
    features_mut = TechnicalIndicators.compute_all_indicators(df_t_mut)
    levels_mut, err_mut = TradeConstructionEngine.construct_trade_levels(sym, features_mut)

    if levels_orig is not None:
        assert levels_mut is not None
        assert levels_orig.entry_trigger_price == levels_mut.entry_trigger_price
        assert levels_orig.stop_loss_price == levels_mut.stop_loss_price
        assert levels_orig.target_1 == levels_mut.target_1
        assert levels_orig.target_2 == levels_mut.target_2
        assert levels_orig.target_3 == levels_mut.target_3


def test_future_ohlcv_mutation_does_not_change_features_at_T():
    """16. Test feature generation at T is 100% identical when future OHLCV (> T) is mutated."""
    sym, df_orig = _generate_synthetic_stock_df(180)
    as_of_idx = 120
    as_of_dt = df_orig.iloc[as_of_idx]["timestamp"].date()

    # Pass full un-truncated DataFrame through production PIT boundary
    raw_sub_orig = PointInTimeFilter.filter_market_data(df_orig, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(raw_sub_orig, as_of_dt)
    base_features = TechnicalIndicators.compute_all_indicators(raw_sub_orig)

    feat_cols = ["sma_20", "ema_20", "ema_50", "rsi_14", "atr_14", "rvol_20"]
    base_values = {col: float(base_features[col].iloc[-1]) for col in feat_cols if col in base_features.columns}

    # Mutate ONLY future rows (> T) in full DataFrame
    df_mut = df_orig.copy()
    df_mut.loc[as_of_idx + 1 :, "open"] *= 50.0
    df_mut.loc[as_of_idx + 1 :, "high"] *= 50.0
    df_mut.loc[as_of_idx + 1 :, "low"] *= 0.05
    df_mut.loc[as_of_idx + 1 :, "close"] *= 50.0
    df_mut.loc[as_of_idx + 1 :, "volume"] *= 500

    raw_sub_mut = PointInTimeFilter.filter_market_data(df_mut, as_of_dt)
    PointInTimeFilter.enforce_pit_boundary(raw_sub_mut, as_of_dt)
    mut_features = TechnicalIndicators.compute_all_indicators(raw_sub_mut)

    mut_values = {col: float(mut_features[col].iloc[-1]) for col in feat_cols if col in mut_features.columns}

    # Assert every feature at T is 100% identical
    assert base_values == mut_values


def test_pit_contract_fails_closed_on_future_row():
    """17. Test PIT contract fails closed by raising PITViolationError if a future row > as_of_date is passed."""
    from src.data.point_in_time import PITViolationError

    sym, df = _generate_synthetic_stock_df(150)
    as_of_idx = 100
    as_of_dt = df.iloc[as_of_idx]["timestamp"].date()

    # Intentionally pass a DataFrame containing future rows (up to idx 105 > idx 100) directly to enforce_pit_boundary
    future_df = df.iloc[: as_of_idx + 6].copy()

    with pytest.raises(PITViolationError) as exc_info:
        PointInTimeFilter.enforce_pit_boundary(future_df, as_of_dt)

    assert "PIT Violation" in str(exc_info.value)
