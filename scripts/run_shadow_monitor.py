#!/usr/bin/env python3
"""
Shadow Monitor Entry Point — scripts/run_shadow_monitor.py

Loads all active shadow trades from database and evaluates EOD price action.
Updates stop/target statuses and prints a performance summary report.

Usage:
  python scripts/run_shadow_monitor.py
  python scripts/run_shadow_monitor.py --date YYYY-MM-DD
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_db_session, init_db
from src.shadow.monitor import ShadowPerformanceReport, ShadowTradeUpdate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_shadow_monitor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Shadow Trade Monitor")
    parser.add_argument("--date", type=str, default=None, help="Monitor date (YYYY-MM-DD)")
    return parser.parse_args()


def main():
    args = parse_args()
    monitor_date = date.fromisoformat(args.date) if args.date else date.today()

    logger.info(f"Shadow Monitor running for: {monitor_date}")
    init_db()

    # In production: load from DB
    # Simulating for demo with sample trades
    mock_trades = [
        {
            "symbol": "TRENT", "status": "ACTIVE", "entry_price": 7200.0,
            "stop_loss": 6900.0, "target_1": 7740.0, "target_2": 8040.0, "target_3": 8550.0,
            "recommendation_id": "REC-TRENT-01", "session_count": 3,
        },
    ]

    mock_eod_bars = {
        "TRENT": {"open": 7350.0, "high": 7800.0, "low": 7300.0, "close": 7750.0}
    }

    updated_trades = []
    for trade in mock_trades:
        bar = mock_eod_bars.get(trade["symbol"])
        if bar:
            updated = ShadowTradeUpdate.check_and_update(trade, bar, trade.get("session_count", 1))
            updated_trades.append(updated)

    closed = [t for t in updated_trades if t.get("status") not in ("ACTIVE",)]
    report = ShadowPerformanceReport.generate_report(closed)

    logger.info(f"\n{'='*50}")
    logger.info(f"SHADOW PORTFOLIO PERFORMANCE SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"  Total Closed : {report['total']}")
    logger.info(f"  Win Rate     : {report['win_rate']:.1f}%")
    logger.info(f"  Avg P&L      : {report['avg_pnl_pct']:+.2f}%")
    logger.info(f"  Total P&L    : {report['total_pnl_pct']:+.2f}%")
    logger.info(f"{'='*50}")

    for t in updated_trades:
        status = t.get("status", "ACTIVE")
        pnl = t.get("pnl_pct")
        pnl_str = f" | P&L: {pnl:+.2f}%" if pnl is not None else ""
        logger.info(f"  [{status}] {t['symbol']}{pnl_str}")


if __name__ == "__main__":
    main()
