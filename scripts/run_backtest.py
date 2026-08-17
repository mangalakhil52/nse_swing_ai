#!/usr/bin/env python3
"""
Backtesting Entry Point — scripts/run_backtest.py (Part 21 & Part 23 Compliance)

Runs genuine historical portfolio walk-forward backtest using real market data from HistoricalDataProvider.
ZERO synthetic price generation (no np.random, no fake candles).

Usage:
  python scripts/run_backtest.py --symbol TRENT --lookback 365
  python scripts/run_backtest.py --all-universe --lookback 365
"""

import argparse
import asyncio
from datetime import date, timedelta
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from src.data.historical_provider import HistoricalDataProvider
from src.data.historical_universe import HistoricalUniverseProvider
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Backtesting Engine")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to backtest")
    parser.add_argument("--lookback", type=int, default=365, help="Historical days to lookback")
    parser.add_argument("--all-universe", action="store_true", help="Backtest all universe symbols")
    return parser.parse_args()


def generate_signals_from_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Generates entry signals from pattern detection on historical bars."""
    enriched = TechnicalIndicators.compute_all_indicators(df)
    signals = []

    for i in range(50, len(enriched) - 15):
        window_df = enriched.iloc[: i + 1].copy()
        patterns = PatternRecognizer.evaluate_all_patterns(window_df)

        for p in patterns:
            if p.is_matched and p.quality_score >= 75.0 and p.breakout_price > 0.0:
                entry_price = p.breakout_price
                sl_price = p.support_stop_price
                risk = entry_price - sl_price

                if risk <= 0 or (risk / entry_price) > 0.08:
                    continue

                signals.append({
                    "entry_idx": i,
                    "entry_price": entry_price,
                    "stop_loss": sl_price,
                    "target_1": entry_price + risk * 1.8,
                    "target_2": entry_price + risk * 2.8,
                    "target_3": entry_price + risk * 4.5,
                    "shares": max(1, int(10000 / entry_price)),
                    "pattern": p.pattern_type.value,
                })
            break

    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["entry_idx", "entry_price", "stop_loss", "target_1", "target_2", "target_3", "shares"]
    )


async def run_single_backtest_async(symbol: str, lookback: int) -> BacktestResult | None:
    logger.info(f"Running backtest for {symbol} using REAL market data (lookback: {lookback} days)...")
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback)

    hist_provider = HistoricalDataProvider()
    try:
        df = await hist_provider.get_daily_ohlcv(symbol, start_dt, end_dt, min_bars=50)
    except Exception as e:
        logger.warning(f"Skipping {symbol} in backtest: {e}")
        return None

    try:
        signal_df = generate_signals_from_df(df, symbol)
        if signal_df.empty:
            logger.info(f"No valid trading signals generated for {symbol}.")
            return None

        result = BacktestEngine.run_strategy_backtest(signal_df, df, symbol)

        logger.info(f"\n{'='*50}")
        logger.info(f"BACKTEST RESULTS: {symbol}")
        logger.info(f"{'='*50}")
        logger.info(f"  Total Trades  : {result.total_trades}")
        logger.info(f"  Win Rate      : {result.win_rate_pct:.1f}%")
        logger.info(f"  Avg Gain      : +{result.avg_gain_pct:.2f}%")
        logger.info(f"  Avg Loss      : {result.avg_loss_pct:.2f}%")
        logger.info(f"  Profit Factor : {result.profit_factor:.2f}")
        logger.info(f"  Expectancy    : ₹{result.expectancy_rupees:.2f} / trade")
        logger.info(f"  Total P&L     : ₹{result.total_pnl_rupees:,.2f}")
        logger.info(f"  Max Drawdown  : {result.max_drawdown_pct:.2f}%")
        logger.info(f"{'='*50}\n")
        return result
    except Exception as e:
        logger.warning(f"Error processing backtest for {symbol}: {e}")
        return None
    return result


async def run_universe_backtest_async(lookback: int) -> None:
    logger.info(f"Running genuine universe-wide portfolio backtest (lookback: {lookback} days)...")
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback)

    # Survivorship-safe historical universe
    symbols = HistoricalUniverseProvider.get_universe_for_date(start_dt)
    logger.info(f"Loaded {len(symbols)} eligible historical universe symbols.")

    all_trades: list[BacktestTrade] = []

    for sym in symbols:
        try:
            res = await run_single_backtest_async(sym, lookback)
            if res and res.trade_details:
                all_trades.extend(res.trade_details)
        except Exception as e:
            logger.debug(f"Universe backtest error for {sym}: {e}")

    summary = BacktestEngine._compute_stats(all_trades)

    logger.info(f"\n{'='*50}")
    logger.info(f"UNIVERSE-WIDE PORTFOLIO BACKTEST SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"  Total Trades  : {summary.total_trades}")
    logger.info(f"  Win Rate      : {summary.win_rate_pct:.1f}%")
    logger.info(f"  Avg Gain      : +{summary.avg_gain_pct:.2f}%")
    logger.info(f"  Avg Loss      : {summary.avg_loss_pct:.2f}%")
    logger.info(f"  Profit Factor : {summary.profit_factor:.2f}")
    logger.info(f"  Expectancy    : ₹{summary.expectancy_rupees:.2f} / trade")
    logger.info(f"  Total P&L     : ₹{summary.total_pnl_rupees:,.2f}")
    logger.info(f"  Max Drawdown  : {summary.max_drawdown_pct:.2f}%")
    logger.info(f"{'='*50}\n")


def main():
    args = parse_args()

    if args.symbol:
        asyncio.run(run_single_backtest_async(args.symbol.upper(), args.lookback))
    elif args.all_universe:
        asyncio.run(run_universe_backtest_async(args.lookback))
    else:
        logger.error("Provide --symbol or --all-universe")
        sys.exit(1)


if __name__ == "__main__":
    main()
