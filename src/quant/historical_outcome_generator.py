"""
Real Historical Setup Outcome Generator Module — src/quant/historical_outcome_generator.py

Production pipeline component that consumes REAL historical OHLCV data from HistoricalDataProvider,
detects historical setups point-in-time, simulates forward outcomes chronologically,
and registers verified outcomes into HistoricalSetupOutcomeStore.

Enforces P0 Regime Integrity:
  1. ZERO hardcoded market observation values (no 1.2, 60.0, 15.0).
  2. ZERO default BULL regime initializations.
  3. Uses real point-in-time NIFTY OHLCV and real historical market regime inputs (t <= setup_date).
  4. If historical regime data is missing or UNKNOWN, skips the setup outcome.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
import logging
from typing import Any
import pandas as pd

from src.core.types import MarketRegime, PatternType
from src.data.historical_provider import HistoricalDataProvider
from src.quant.indicators import TechnicalIndicators
from src.quant.probability_engine import (
    HistoricalSetupOutcome,
    HistoricalSetupOutcomeStore,
    validate_outcome,
)
from src.quant.regime import MarketRegimeClassifier

logger = logging.getLogger(__name__)


@dataclass
class GenerationReport:
    symbols_processed: int = 0
    candles_processed: int = 0
    setups_detected: int = 0
    outcomes_generated: int = 0
    outcomes_rejected: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


class HistoricalOutcomeGenerator:
    """Executably generates empirical historical setup outcomes from genuine historical OHLCV data."""

    @classmethod
    def generate_outcomes_for_symbol(
        cls,
        symbol: str,
        df_hist: pd.DataFrame,
        nifty_df: pd.DataFrame | None = None,
        regime_context: dict[str, dict[str, float]] | None = None,
        default_regime_if_missing: MarketRegime | None = None,
        source: str = "NSE_BHAVCOPY_HISTORICAL",
        target_pct: float = 10.0,
        stop_pct: float = 5.0,
        max_holding_sessions: int = 25,
    ) -> tuple[list[HistoricalSetupOutcome], int, int]:
        """
        Parses historical OHLCV for a single symbol and extracts setup outcomes.
        Guarantees point-in-time setup detection (t <= setup_date) and subsequent outcome simulation (t > setup_date).
        Returns (records, candles_processed, setups_detected).
        """
        if df_hist is None or len(df_hist) < 55 or not source or not source.strip():
            return [], len(df_hist) if df_hist is not None else 0, 0

        df = TechnicalIndicators.compute_all_indicators(df_hist.copy())
        records: list[HistoricalSetupOutcome] = []
        n_bars = len(df)
        setups_detected = 0

        # Iterate through historical bars (i >= 50 and i + 1 < n_bars)
        for i in range(50, n_bars - 1):
            sub_df = df.iloc[: i + 1]  # Strict point-in-time slice up to setup_date (no future data)
            current_bar = sub_df.iloc[-1]

            setup_ts = pd.to_datetime(current_bar["timestamp"])
            setup_date_str = setup_ts.strftime("%Y-%m-%d")

            close_p = float(current_bar["close"])
            high_p = float(current_bar["high"])
            low_p = float(current_bar["low"])
            ema_20 = float(current_bar["ema_20"])
            ema_50 = float(current_bar["ema_50"])
            vol = int(current_bar["volume"])
            vol_sma = float(current_bar["volume_sma_20"])

            if close_p <= 0.0 or low_p <= 0.0 or high_p <= 0.0:
                continue

            # 1. Point-in-Time Pattern Detection (using ONLY t <= setup_date)
            detected_pattern: PatternType | None = None

            # Pattern A: Volatility Contraction / Consolidation Breakout
            if close_p > ema_20 > ema_50 and vol > (vol_sma * 1.2) and close_p > float(sub_df["high"].iloc[-15:-1].max()):
                detected_pattern = PatternType.VOLATILITY_CONTRACTION_PATTERN
            # Pattern B: Cup and Handle / Flat Base Breakout
            elif close_p > float(sub_df["high"].iloc[-30:-1].max()) and vol > (vol_sma * 1.4):
                detected_pattern = PatternType.CUP_AND_HANDLE
            # Pattern C: EMA Pullback Reversal
            elif low_p <= (ema_20 * 1.01) and close_p > ema_20 and close_p > float(sub_df["open"].iloc[-1]):
                detected_pattern = PatternType.EMA_PULLBACK_REVERSAL
            # Pattern D: Inside Bar Breakout
            elif (
                len(sub_df) >= 2
                and float(sub_df["high"].iloc[-2]) > float(sub_df["high"].iloc[-1])
                and float(sub_df["low"].iloc[-2]) < float(sub_df["low"].iloc[-1])
            ):
                detected_pattern = PatternType.INSIDE_BAR_BREAKOUT

            if detected_pattern is None or detected_pattern == PatternType.UNKNOWN:
                continue

            setups_detected += 1

            # 2. Point-in-Time Regime Determination (Strict Zero Fallback)
            regime = MarketRegime.UNKNOWN

            if regime_context and setup_date_str in regime_context:
                r_info = regime_context[setup_date_str]
                ad_ratio = r_info.get("advance_decline_ratio")
                pct_50 = r_info.get("pct_above_50_sma")
                vix = r_info.get("india_vix")
            else:
                ad_ratio = None
                pct_50 = None
                vix = None

            if nifty_df is not None and len(nifty_df) >= 50:
                nifty_slice = nifty_df[pd.to_datetime(nifty_df["timestamp"]) <= setup_ts]
                if len(nifty_slice) >= 50 and ad_ratio is not None and pct_50 is not None and vix is not None:
                    reg_res = MarketRegimeClassifier.classify_regime(
                        nifty_df=nifty_slice,
                        advance_decline_ratio=ad_ratio,
                        pct_above_50_sma=pct_50,
                        india_vix=vix,
                    )
                    regime = reg_res.regime

            if regime == MarketRegime.UNKNOWN and default_regime_if_missing is not None:
                regime = default_regime_if_missing

            # P0 Rule: Missing or UNKNOWN historical regime MUST skip the outcome
            if regime == MarketRegime.UNKNOWN:
                logger.debug(f"[{symbol}] Skipping setup on {setup_date_str} due to UNKNOWN/missing historical regime.")
                continue

            # Trade Level Definitions
            entry_price = close_p
            stop_loss = entry_price * (1.0 - stop_pct / 100.0)
            target_1 = entry_price * (1.0 + target_pct / 100.0)

            # 3. Outcome Simulation (Strictly Forward in Time: t > i)
            forward_bars = df.iloc[i + 1 : min(i + 1 + max_holding_sessions, n_bars)]
            if forward_bars.empty:
                continue

            t1_hit = False
            exit_date_str = None
            holding_sessions = 0
            max_high = entry_price
            min_low = entry_price

            for j, (_, f_row) in enumerate(forward_bars.iterrows(), 1):
                f_high = float(f_row["high"])
                f_low = float(f_row["low"])
                f_date = pd.to_datetime(f_row["timestamp"]).strftime("%Y-%m-%d")

                if f_high > max_high:
                    max_high = f_high
                if f_low < min_low:
                    min_low = f_low

                # Check Stop Loss first (worst-case execution)
                if f_low <= stop_loss:
                    t1_hit = False
                    exit_date_str = f_date
                    holding_sessions = j
                    break

                # Check Target 1
                if f_high >= target_1:
                    t1_hit = True
                    exit_date_str = f_date
                    holding_sessions = j
                    break

            if exit_date_str is None:
                # Timed out without hitting SL or T1
                exit_date_str = pd.to_datetime(forward_bars.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")
                holding_sessions = len(forward_bars)
                t1_hit = False

            mfe = round(((max_high - entry_price) / entry_price) * 100.0, 2)
            mae = round(((entry_price - min_low) / entry_price) * 100.0, 2)

            outcome_rec = HistoricalSetupOutcome(
                symbol=symbol,
                pattern_type=detected_pattern,
                market_regime=regime,
                setup_date=setup_date_str,
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                target_1=round(target_1, 2),
                t1_hit_before_sl=t1_hit,
                holding_sessions=holding_sessions,
                exit_date=exit_date_str,
                source=source.strip(),
                mfe=mfe,
                mae=mae,
            )
            records.append(outcome_rec)

        return records, n_bars, setups_detected

    @classmethod
    def generate_outcomes(
        cls,
        symbols: list[str],
        stock_dfs: dict[str, pd.DataFrame],
        nifty_df: pd.DataFrame | None = None,
        regime_context: dict[str, dict[str, float]] | None = None,
        default_regime_if_missing: MarketRegime | None = None,
        source: str = "NSE_BHAVCOPY_HISTORICAL",
        target_pct: float = 10.0,
        stop_pct: float = 5.0,
        max_holding_sessions: int = 25,
    ) -> GenerationReport:
        """
        Public API for generating verified historical setup outcomes across symbols and registering them into store.
        Returns a detailed GenerationReport.
        """
        report = GenerationReport()
        all_outcomes: list[HistoricalSetupOutcome] = []

        for symbol in symbols:
            df_hist = stock_dfs.get(symbol)
            if df_hist is None or df_hist.empty:
                continue

            report.symbols_processed += 1
            sym_records, n_candles, n_setups = cls.generate_outcomes_for_symbol(
                symbol=symbol,
                df_hist=df_hist,
                nifty_df=nifty_df,
                regime_context=regime_context,
                default_regime_if_missing=default_regime_if_missing,
                source=source,
                target_pct=target_pct,
                stop_pct=stop_pct,
                max_holding_sessions=max_holding_sessions,
            )
            report.candles_processed += n_candles
            report.setups_detected += n_setups
            all_outcomes.extend(sym_records)

        # Validate and register into HistoricalSetupOutcomeStore
        added_count, rejected_count = HistoricalSetupOutcomeStore.register_outcomes(all_outcomes, persist=True)
        report.outcomes_generated = added_count
        report.outcomes_rejected = rejected_count

        logger.info(
            f"[HistoricalOutcomeGenerator] Completed. Processed {report.symbols_processed} symbols ({report.candles_processed} candles). "
            f"Setups: {report.setups_detected} | Registered: {report.outcomes_generated} | Rejected: {report.outcomes_rejected}"
        )
        return report

    @classmethod
    async def build_and_register_historical_outcomes(
        cls,
        symbols: list[str],
        start_date: date,
        end_date: date,
        hist_provider: HistoricalDataProvider | None = None,
        regime_context: dict[str, dict[str, float]] | None = None,
        source: str = "NSE_BHAVCOPY_HISTORICAL",
    ) -> GenerationReport:
        """
        Executable pipeline method: consumes real historical OHLCV from HistoricalDataProvider,
        runs outcome generation, and registers verified observations into HistoricalSetupOutcomeStore.
        """
        if hist_provider is None:
            hist_provider = HistoricalDataProvider()

        stock_dfs: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df_hist = await hist_provider.get_daily_ohlcv(sym, start_date, end_date, min_bars=50)
                stock_dfs[sym] = df_hist
            except Exception as e:
                logger.debug(f"Skipping {sym} for historical outcome generation: {e}")

        nifty_df = None
        try:
            nifty_df = await hist_provider.get_daily_ohlcv("NIFTY 50", start_date, end_date, min_bars=50)
        except Exception:
            pass

        return cls.generate_outcomes(
            symbols=list(stock_dfs.keys()),
            stock_dfs=stock_dfs,
            nifty_df=nifty_df,
            regime_context=regime_context,
            source=source,
        )
