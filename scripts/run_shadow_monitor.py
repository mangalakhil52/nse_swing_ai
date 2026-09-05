#!/usr/bin/env python3
"""
Shadow Monitor Entry Point — scripts/run_shadow_monitor.py

Loads active shadow trades from SQLite database, fetches real EOD market bars,
evaluates stop/target triggers via ShadowTradeUpdate state machine,
persists updated positions to database, and outputs analytical performance reports.

Usage:
  python scripts/run_shadow_monitor.py
  python scripts/run_shadow_monitor.py --date YYYY-MM-DD
"""

import argparse
import asyncio
from datetime import date, timedelta
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.historical_provider import HistoricalDataProvider
from src.database.connection import get_db_session, init_db
from src.database.schema import ShadowTradeModel
from src.shadow.analyzer import RealTradeAnalyzer
from src.shadow.monitor import ShadowTradeUpdate

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


async def run_shadow_monitor_async(monitor_date: date) -> int:
    logger.info("=" * 60)
    logger.info(f"NSE SWING AI — SHADOW PORTFOLIO MONITORING ({monitor_date})")
    logger.info("=" * 60)

    init_db()

    with get_db_session() as session:
        # Load active trades from database
        active_db_trades = (
            session.query(ShadowTradeModel)
            .filter(ShadowTradeModel.status == "ACTIVE")
            .all()
        )

        # Seed initial paper trade if database is empty for testing/first run
        if not active_db_trades and session.query(ShadowTradeModel).count() == 0:
            logger.info("Database contains 0 shadow trades. Seeding initial baseline position (TRENT)...")
            initial_trade = ShadowTradeModel(
                recommendation_id="REC-TRENT-01",
                symbol="TRENT",
                entry_date=monitor_date - timedelta(days=3),
                entry_price=7200.0,
                stop_loss=6900.0,
                target_1=7740.0,
                target_2=8040.0,
                target_3=8550.0,
                position_size_shares=50,
                status="ACTIVE",
                holding_sessions=1,
            )
            session.add(initial_trade)
            session.commit()
            active_db_trades = [initial_trade]

        logger.info(f"Loaded {len(active_db_trades)} active shadow positions from database.")

        if not active_db_trades:
            logger.info("No active positions to evaluate.")
            metrics = RealTradeAnalyzer.analyze_trades_from_db(session)
            _print_summary(metrics)
            return 0

        # Fetch real EOD market data for active symbols
        hist_provider = HistoricalDataProvider()
        for trade_model in active_db_trades:
            try:
                sym = trade_model.symbol
                session_cnt = (trade_model.holding_sessions or 1) + 1

                df_bar = await hist_provider.get_daily_ohlcv(sym, monitor_date - timedelta(days=10), monitor_date, min_bars=1)
                if df_bar.empty:
                    logger.warning(f"No EOD price bar available for {sym} on {monitor_date}.")
                    continue

                last_row = df_bar.iloc[-1]
                bar_dict = {
                    "open": float(last_row["open"]),
                    "high": float(last_row["high"]),
                    "low": float(last_row["low"]),
                    "close": float(last_row["close"]),
                }

                trade_dict = {
                    "symbol": trade_model.symbol,
                    "entry_price": trade_model.entry_price,
                    "stop_loss": trade_model.stop_loss or (trade_model.entry_price * 0.95),
                    "target_1": trade_model.target_1 or (trade_model.entry_price * 1.05),
                    "target_2": trade_model.target_2 or (trade_model.entry_price * 1.10),
                    "target_3": trade_model.target_3 or (trade_model.entry_price * 1.15),
                    "t1_hit": trade_model.t1_hit,
                    "t2_hit": trade_model.t2_hit,
                    "t3_hit": trade_model.t3_hit,
                    "status": trade_model.status,
                }

                updated = ShadowTradeUpdate.check_and_update(trade_dict, bar_dict, session_cnt)

                # Persist state changes back to database model
                trade_model.holding_sessions = session_cnt
                trade_model.t1_hit = updated.get("t1_hit", trade_model.t1_hit)
                trade_model.t2_hit = updated.get("t2_hit", trade_model.t2_hit)
                trade_model.t3_hit = updated.get("t3_hit", trade_model.t3_hit)

                if updated.get("status") != "ACTIVE":
                    trade_model.status = updated["status"]
                    trade_model.exit_price = updated.get("exit_price")
                    trade_model.exit_date = monitor_date
                    trade_model.exit_reason = updated.get("exit_reason")
                    trade_model.pnl_percentage = updated.get("pnl_pct", 0.0)

                    shares = trade_model.position_size_shares or 100
                    pnl_per_share = (trade_model.exit_price - trade_model.entry_price) if trade_model.exit_price else 0.0
                    trade_model.pnl_rupees = round(pnl_per_share * shares, 2)

                    logger.info(f"Position Closed: [{trade_model.symbol}] Status={trade_model.status} | Exit=₹{trade_model.exit_price} | P&L={trade_model.pnl_percentage:+.2f}%")

            except Exception as e:
                logger.warning(f"Skipping trade update for {getattr(trade_model, 'symbol', 'UNKNOWN')}: {e}")

        session.commit()

        # Compute analytical summary
        metrics = RealTradeAnalyzer.analyze_trades_from_db(session)
        _print_summary(metrics)

    return 0


def _print_summary(metrics):
    logger.info(f"\n{'='*60}")
    logger.info(f"REAL/SHADOW TRADE PERFORMANCE ANALYTICS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Total Trades Analyzed : {metrics.total_trades}")
    logger.info(f"  Wins / Losses         : {metrics.wins} / {metrics.losses}")
    logger.info(f"  Win Rate              : {metrics.win_rate_pct:.1f}%")
    logger.info(f"  Gross Profit          : ₹{metrics.gross_profit_rupees:,.2f}")
    logger.info(f"  Gross Loss            : ₹{metrics.gross_loss_rupees:,.2f}")
    logger.info(f"  Profit Factor         : {metrics.profit_factor:.2f}")
    logger.info(f"  Trade Expectancy      : ₹{metrics.expectancy_rupees:,.2f} / trade")
    logger.info(f"  Total Net P&L         : ₹{metrics.total_pnl_rupees:,.2f} ({metrics.total_pnl_pct:+.2f}%)")
    logger.info(f"  Max Drawdown          : {metrics.max_drawdown_pct:.2f}%")
    logger.info(f"  Avg Holding Period    : {metrics.avg_holding_sessions:.1f} sessions")
    logger.info(f"  Exit Breakdown        : {metrics.exit_reasons}")
    logger.info(f"{'='*60}\n")


def main():
    args = parse_args()
    monitor_date = date.fromisoformat(args.date) if args.date else date.today()
    exit_code = asyncio.run(run_shadow_monitor_async(monitor_date))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
