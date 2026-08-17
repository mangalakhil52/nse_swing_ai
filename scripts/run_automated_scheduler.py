#!/usr/bin/env python3
"""
Automated 5:00 PM Daily Scheduler — scripts/run_automated_scheduler.py

Runs automatically every trading day at 17:00 IST (5:00 PM):
  1. Verifies NSE trading day schedule (skips weekends and holidays).
  2. Downloads official EOD Bhavcopy and runs full CIO multi-agent daily scan.
  3. Appends new trade recommendations to Excel Ledger (data/trade_ledger.xlsx).
  4. Evaluates EOD price action against open positions in data/trade_ledger.xlsx
     and auto-updates status (Pending -> Active, Target Hit, Stop Loss, Time Expired, P&L %).
  5. Dispatches Telegram alerts for new recommendations and daily ledger updates.

Usage:
  python scripts/run_automated_scheduler.py [--run-now] [--setup-windows-task]
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_hours import MarketCalendar
from config.settings import settings
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.models import SymbolMetadata
from src.database.connection import get_db_session, init_db
from src.database.repository import DatabaseRepository
from src.quant.regime import MarketRegimeClassifier
from src.quant.screener import QuantScreener
from src.shadow.excel_ledger import TradeLedgerExcelManager
from src.shadow.telegram_bot import TelegramBotNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_automated_scheduler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Automated 5 PM Scheduler")
    parser.add_argument("--run-now", action="store_true", help="Execute scan and ledger update immediately")
    parser.add_argument("--setup-windows-task", action="store_true", help="Register automatic daily 5 PM Windows Task")
    parser.add_argument("--date", type=str, default=None, help="Override target date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Force scan execution even on weekends/holidays")
    return parser.parse_args()


async def execute_daily_5pm_cycle(target_date: date | None = None, force: bool = False) -> int:
    """Executes the complete 5:00 PM automated trading cycle."""
    target_date = MarketCalendar.get_latest_trading_day(target_date or date.today())
    run_id = f"AUTO-5PM-{target_date.strftime('%Y%m%d')}-{int(datetime.now().timestamp())}"

    logger.info(f"{'='*60}")
    logger.info(f"NSE SWING AI — AUTOMATED 5:00 PM DAILY CYCLE")
    logger.info(f"Run ID: {run_id} | Date: {target_date} | Force: {force}")
    logger.info(f"{'='*60}")

    # 1. Trading Day Check
    if not force and not MarketCalendar.is_trading_day(target_date):
        logger.info(f"{target_date} is a weekend or NSE market holiday. Skipping automated scan. Use --force to override.")
        return 0

    # 2. Initialize DB & Providers
    init_db()
    from src.data.nse_provider import NseDataProvider
    from src.data.universe import UniverseDiscoveryEngine

    nse_provider = NseDataProvider()
    telegram_bot = TelegramBotNotifier()
    ledger_manager = TradeLedgerExcelManager()

    # 3. Build Universe
    logger.info("Updating NSE universe master...")
    universe_engine = UniverseDiscoveryEngine(market_data_provider=nse_provider)
    universe_meta = await universe_engine.build_universe()
    universe_dict: dict[str, SymbolMetadata] = {u.symbol: u for u in universe_meta}

    # 4. Fetch EOD Bhavcopy
    logger.info(f"Fetching official EOD Bhavcopy for {target_date}...")
    bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(target_date)
    if bhavcopy_df.empty:
        logger.error(f"Bhavcopy data not available for {target_date}. Will retry on next cycle.")
        await nse_provider.close()
        return 1

    # 5. Build stock time-series DataFrames
    import pandas as pd
    import numpy as np

    bhavcopy_prices = dict(zip(bhavcopy_df["symbol"], bhavcopy_df["close"]))
    bhavcopy_vols = dict(zip(bhavcopy_df["symbol"], bhavcopy_df["volume"]))
    bhavcopy_dels = dict(zip(bhavcopy_df["symbol"], bhavcopy_df.get("delivery_pct", pd.Series(50.0))))

    stock_dfs: dict[str, pd.DataFrame] = {}
    # 5. Load real historical OHLCV history for each symbol (P0.1 & P0.2)
    from src.data.historical_provider import HistoricalDataProvider

    hist_provider = HistoricalDataProvider()
    start_history_date = target_date - pd.Timedelta(days=120)

    for sym_meta in universe_meta:
        sym = sym_meta.symbol
        try:
            df_hist = await hist_provider.get_daily_ohlcv(sym, start_history_date, target_date, min_bars=50)
            stock_dfs[sym] = df_hist
        except Exception as e:
            logger.debug(f"Skipping {sym} due to unavailable/insufficient historical data: {e}")

    # 6. Market Regime Classification (Real NIFTY OHLCV Data)
    nifty_df = pd.DataFrame()
    try:
        nifty_df = await hist_provider.get_daily_ohlcv("NIFTY 50", start_history_date, target_date, min_bars=50)
    except Exception as e:
        logger.warning(f"DATA_UNAVAILABLE: Could not fetch NIFTY 50 data ({e}). Market regime set to UNKNOWN.")

    regime_result = MarketRegimeClassifier.classify_regime(nifty_df=nifty_df)

    if not regime_result.allow_long_swing_trades:
        logger.warning(f"Market regime {regime_result.regime.value} prohibits long trades. Completing with 0 recommendations.")
        return

    # 7. Stage-1 Screener & CIO Multi-Agent Research
    logger.info("Executing Stage-1 Quant Screener & Multi-Agent Research...")
    screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
    candidates = screener.screen_universe(universe_meta, stock_dfs, nifty_df)

    cio = CIOOrchestrator()
    recommendations = await cio.run_daily_scan(
        candidates=candidates,
        stock_dfs=stock_dfs,
        universe=universe_dict,
        regime_result=regime_result,
        run_id=run_id,
    )

    # 8. Record new recommendations in Excel Ledger
    if recommendations:
        added = ledger_manager.record_recommendations(recommendations)
        logger.info(f"Recorded {added} new trade recommendations in data/trade_ledger.xlsx")

        # Save to database repository
        with get_db_session() as session:
            repo = DatabaseRepository(session)
            repo.create_agent_run(
                run_id=run_id,
                market_regime=regime_result.regime.value,
                universe_size=len(universe_meta),
                quant_candidates_count=len(candidates),
            )
            for rec in recommendations:
                repo.save_trade_recommendation(rec)
            repo.complete_agent_run(run_id, "COMPLETED", len(recommendations))

    # 9. Update Open Trades in Excel Ledger using EOD Bhavcopy
    logger.info("Updating status and P&L for all open trades in Excel ledger...")
    update_summary = ledger_manager.update_ledger_with_eod_data(bhavcopy_df, target_date)
    logger.info(f"Ledger Update Summary:\n{update_summary}")

    # 10. Dispatch Telegram Notifications
    if telegram_bot.is_configured:
        logger.info("Dispatching Telegram notifications...")
        await telegram_bot.dispatch_recommendations(recommendations, regime_result.regime.value)
        await telegram_bot.dispatch_ledger_update(update_summary)

    await nse_provider.close()
    logger.info(f"✅ Automated 5:00 PM cycle completed successfully for {target_date}.")
    return 0


def setup_windows_task_scheduler() -> None:
    """Creates an automated 5:00 PM IST daily Windows Task Scheduler job."""
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())
    task_name = "NSESwingAI_Daily_5PM_Scan"

    cmd = (
        f'schtasks /create /tn "{task_name}" /tr "\"{python_exe}\" \"{script_path}\" --run-now" '
        f'/sc daily /st 17:00 /f'
    )
    logger.info(f"Registering Windows Task Scheduler job: {task_name} at 17:00 IST...")
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            logger.info(f"✅ Successfully registered Windows Task Scheduler job '{task_name}' for 17:00 IST daily!")
        else:
            logger.error(f"Failed to register Windows task: {res.stderr}")
    except Exception as e:
        logger.error(f"Exception registering Task Scheduler job: {e}")


def run_continuous_loop() -> None:
    """Continuous loop mode checking time every minute and running at 17:00 IST."""
    logger.info("Continuous scheduler active. Listening for daily 17:00 IST trigger...")
    last_run_date = None

    while True:
        now = datetime.now()
        # Trigger at 17:00 (5:00 PM) on trading days
        if now.hour == 17 and now.minute == 0 and last_run_date != now.date():
            logger.info("Clock reached 17:00 IST. Initiating automated daily cycle...")
            asyncio.run(execute_daily_5pm_cycle(now.date()))
            last_run_date = now.date()

        time.sleep(30)


def main():
    args = parse_args()

    if args.setup_windows_task:
        setup_windows_task_scheduler()
        return

    if args.run_now or args.date:
        target = date.fromisoformat(args.date) if args.date else date.today()
        exit_code = asyncio.run(execute_daily_5pm_cycle(target, force=args.force))
        sys.exit(exit_code)

    # Default to continuous loop
    run_continuous_loop()


if __name__ == "__main__":
    main()
