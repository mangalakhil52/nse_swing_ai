#!/usr/bin/env python3
"""
Backtesting Entry Point — scripts/run_backtest.py

Runs the historical walk-forward backtest engine with realistic Indian market friction.

Usage:
  python scripts/run_backtest.py --symbol TRENT --lookback 365
  python scripts/run_backtest.py --all-universe --lookback 500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer
from src.quant.screener import QuantScreener

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Backtesting Engine")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to backtest")
    parser.add_argument("--lookback", type=int, default=365, help="Historical bars to use")
    parser.add_argument("--all-universe", action="store_true", help="Backtest all universe symbols")
    return parser.parse_args()


def generate_signals_from_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Generates entry signals from pattern detection on historical bars."""
    enriched = TechnicalIndicators.compute_all_indicators(df)
    signals = []

    for i in range(50, len(enriched) - 15):
        window_df = enriched.iloc[:i + 1].copy()
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
            break  # One signal per bar maximum

    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["entry_idx", "entry_price", "stop_loss", "target_1", "target_2", "target_3", "shares"]
    )


def run_single_backtest(symbol: str, lookback: int) -> None:
    logger.info(f"Running backtest for {symbol} (lookback: {lookback} bars)...")

    # Generate synthetic OHLCV for demo (real impl loads from DB)
    np.random.seed(42)
    n = lookback
    close = np.cumprod(1.0 + np.random.normal(0.0008, 0.018, n)) * 1000.0
    high = close * (1.0 + np.abs(np.random.normal(0.005, 0.01, n)))
    low = close * (1.0 - np.abs(np.random.normal(0.005, 0.01, n)))
    open_p = (high + low) / 2.0
    volume = np.random.randint(200000, 1000000, n)

    df = pd.DataFrame({"open": open_p, "high": high, "low": low, "close": close, "volume": volume})

    signal_df = generate_signals_from_df(df, symbol)
    if signal_df.empty:
        logger.info(f"No signals generated for {symbol}.")
        return

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


def main():
    args = parse_args()

    if args.symbol:
        run_single_backtest(args.symbol.upper(), args.lookback)
    elif args.all_universe:
        logger.info("Universe-wide backtest not yet implemented. Run with --symbol for single stock.")
    else:
        logger.error("Provide --symbol or --all-universe")
        sys.exit(1)


if __name__ == "__main__":
    main()
