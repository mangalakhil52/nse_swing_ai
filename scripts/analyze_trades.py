#!/usr/bin/env python3
"""
Real Trade Analyzer CLI Tool — scripts/analyze_trades.py

Analyzes real and shadow trades from the SQLite database repository or trade ledger CSV files.
Computes Win Rate, Profit Factor, Expectancy, Total Net P&L, Max Drawdown, Average Hold Duration,
and exit reason breakdowns.

Usage:
  python scripts/analyze_trades.py
  python scripts/analyze_trades.py --source db
  python scripts/analyze_trades.py --source csv --ledger-file data/trade_ledger.csv
  python scripts/analyze_trades.py --symbol TRENT
  python scripts/analyze_trades.py --output-json data/trade_analysis.json
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_db_session, init_db
from src.shadow.analyzer import RealTradeAnalyzer, TradeMetricsSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_trades")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Real Trade Analyzer CLI")
    parser.add_argument("--source", type=str, choices=["db", "csv"], default="db", help="Data source: 'db' or 'csv'")
    parser.add_argument("--ledger-file", type=str, default="data/trade_ledger.csv", help="Path to trade ledger CSV file (if source=csv)")
    parser.add_argument("--symbol", type=str, default=None, help="Filter analytics for a specific ticker symbol")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save summary analytics JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"NSE SWING AI — REAL TRADE ANALYZER")
    logger.info(f"Source: {args.source.upper()} | Symbol Filter: {args.symbol or 'ALL'}")
    logger.info("=" * 60)

    if args.source == "db":
        init_db()
        with get_db_session() as session:
            metrics = RealTradeAnalyzer.analyze_trades_from_db(session, symbol=args.symbol)
    else:
        ledger_path = Path(args.ledger_file)
        if not ledger_path.exists():
            logger.error(f"Ledger CSV file not found: {ledger_path}")
            sys.exit(1)

        df = pd.read_csv(ledger_path)
        if args.symbol:
            df = df[df["symbol"].str.upper() == args.symbol.upper().strip()]

        metrics = RealTradeAnalyzer.analyze_trades_from_df(df)

    _print_analysis_report(metrics)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_dict = {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate_pct": metrics.win_rate_pct,
            "gross_profit_rupees": metrics.gross_profit_rupees,
            "gross_loss_rupees": metrics.gross_loss_rupees,
            "profit_factor": metrics.profit_factor,
            "expectancy_rupees": metrics.expectancy_rupees,
            "total_pnl_rupees": metrics.total_pnl_rupees,
            "total_pnl_pct": metrics.total_pnl_pct,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "avg_holding_sessions": metrics.avg_holding_sessions,
            "exit_reasons": metrics.exit_reasons,
        }
        out_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        logger.info(f"Saved analytical summary JSON to: {out_path}")


def _print_analysis_report(metrics: TradeMetricsSummary):
    logger.info(f"\n{'='*60}")
    logger.info(f"REAL TRADE PERFORMANCE METRICS REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"  Total Trades Analyzed : {metrics.total_trades}")
    logger.info(f"  Winning Trades        : {metrics.wins}")
    logger.info(f"  Losing Trades         : {metrics.losses}")
    logger.info(f"  Win Rate (%)          : {metrics.win_rate_pct:.1f}%")
    logger.info(f"  Gross Profit (₹)      : ₹{metrics.gross_profit_rupees:,.2f}")
    logger.info(f"  Gross Loss (₹)        : ₹{metrics.gross_loss_rupees:,.2f}")
    logger.info(f"  Profit Factor         : {metrics.profit_factor:.2f}")
    logger.info(f"  Expectancy (₹/trade)  : ₹{metrics.expectancy_rupees:,.2f}")
    logger.info(f"  Total Realized P&L    : ₹{metrics.total_pnl_rupees:,.2f} ({metrics.total_pnl_pct:+.2f}%)")
    logger.info(f"  Max Drawdown (%)      : {metrics.max_drawdown_pct:.2f}%")
    logger.info(f"  Avg Holding Period    : {metrics.avg_holding_sessions:.1f} sessions")
    logger.info(f"  Exit Reasons Count    : {metrics.exit_reasons}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
