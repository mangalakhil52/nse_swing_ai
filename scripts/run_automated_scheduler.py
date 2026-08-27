#!/usr/bin/env python3
"""Production 17:00 IST scheduler for the NSE Swing AI research pipeline."""

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_hours import MarketCalendar
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.exceptions import DataUnavailableException
from src.database.connection import get_db_session, init_db
from src.database.repository import DatabaseRepository
from src.data.bulk_history import BulkHistoricalLoader
from src.data.nse_index_provider import NseIndexDataProvider
from src.data.nse_provider import NseDataProvider
from src.quant.regime import MarketRegimeClassifier
from src.quant.regime_inputs import compute_breadth, latest_vix
from src.quant.screener import QuantScreener
from src.shadow.excel_ledger import TradeLedgerExcelManager
from src.shadow.telegram_bot import TelegramBotNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_automated_scheduler")
IST = ZoneInfo("Asia/Kolkata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Automated 5 PM Scheduler")
    parser.add_argument("--run-now", action="store_true")
    parser.add_argument("--setup-windows-task", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Override trading-day guard")
    return parser.parse_args()


async def execute_daily_5pm_cycle(target_date: date | None = None, force: bool = False) -> int:
    target_date = target_date or MarketCalendar.get_latest_trading_day(date.today())
    run_id = f"AUTO-5PM-{target_date:%Y%m%d}-{int(datetime.now(IST).timestamp())}"
    logger.info("NSE SWING AI 17:00 cycle | run=%s | date=%s | force=%s", run_id, target_date, force)

    if not force and not MarketCalendar.is_trading_day(target_date):
        logger.info("%s is not an NSE trading day; skipping.", target_date)
        return 0

    init_db()
    nse_provider = NseDataProvider()
    index_provider = NseIndexDataProvider()
    bulk_loader = BulkHistoricalLoader(nse_provider)
    telegram_bot = TelegramBotNotifier()
    ledger_manager = TradeLedgerExcelManager()

    try:
        universe_meta = await nse_provider.fetch_active_securities()
        if not universe_meta:
            raise DataUnavailableException("NSE security master is unavailable")
        universe_dict = {u.symbol: u for u in universe_meta}
        logger.info("Canonical NSE universe: %d securities", len(universe_meta))

        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(target_date)
        if bhavcopy_df.empty:
            raise DataUnavailableException(f"Bhavcopy unavailable for {target_date}")

        # Same-day gate determines which stocks enter the specialist pipeline.
        # It does NOT define market breadth: breadth remains whole-universe.
        bhavcopy_df["_symbol"] = bhavcopy_df["symbol"].astype(str).str.strip().str.upper()
        by_symbol = bhavcopy_df.set_index("_symbol")
        candidate_meta = []
        for meta in universe_meta:
            if meta.symbol not in by_symbol.index:
                continue
            row = by_symbol.loc[meta.symbol]
            if getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            close = float(row.get("close", 0.0))
            turnover = float(row.get("turnover_crores", 0.0))
            if close >= 20.0 and turnover >= 3.0:
                candidate_meta.append(meta)
        logger.info("Same-day gate retained %d/%d symbols", len(candidate_meta), len(universe_meta))

        start_history_date = target_date - timedelta(days=400)
        stock_dfs = await bulk_loader.load(
            [m.symbol for m in universe_meta],
            start_history_date,
            target_date,
            min_bars=100,
        )
        eligible_meta = [m for m in candidate_meta if m.symbol in stock_dfs]
        logger.info("Historical PIT data available for %d/%d NSE symbols", len(stock_dfs), len(universe_meta))

        nifty_df = await index_provider.get_index_history("NIFTY 50", start_history_date, target_date)
        if len(nifty_df) < 200:
            raise DataUnavailableException(f"NIFTY 50 history insufficient: {len(nifty_df)} bars")
        vix_df = await index_provider.get_india_vix_history(start_history_date, target_date)
        ad_ratio, pct_above_50 = compute_breadth(stock_dfs, target_date)
        vix = latest_vix(vix_df, target_date)
        regime_result = MarketRegimeClassifier.classify_regime(
            nifty_df=nifty_df,
            advance_decline_ratio=ad_ratio,
            pct_above_50_sma=pct_above_50,
            india_vix=vix,
            as_of_date=target_date,
        )
        logger.info(
            "Regime=%s | stance=%s | A/D=%.2f | >50SMA=%.1f%% | VIX=%.2f",
            regime_result.regime.value,
            regime_result.trading_stance.value,
            ad_ratio,
            pct_above_50,
            vix,
        )
        if not regime_result.allow_long_swing_trades:
            logger.warning("Regime blocks long swing trades; NO TRADE TODAY")
            return 0

        candidate_dfs = {m.symbol: stock_dfs[m.symbol] for m in eligible_meta if m.symbol in stock_dfs}
        screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
        candidates = screener.screen_universe(eligible_meta, candidate_dfs, nifty_df)
        logger.info("Stage-1 candidates: %d", len(candidates))
        if not candidates:
            logger.info("NO TRADE TODAY: Stage-1 produced no candidates")
            return 0

        cio = CIOOrchestrator()
        recommendations = await cio.run_daily_scan(
            candidates=candidates,
            stock_dfs=candidate_dfs,
            universe={m.symbol: m for m in eligible_meta},
            regime_result=regime_result,
            run_id=run_id,
        )
        logger.info("CIO returned %d recommendations", len(recommendations))

        if recommendations:
            added = ledger_manager.record_recommendations(recommendations)
            logger.info("Recorded %d new recommendations in Excel ledger", added)

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
        else:
            logger.info("NO TRADE TODAY: CIO rejected all candidates")

        update_summary = ledger_manager.update_ledger_with_eod_data(bhavcopy_df, target_date)
        if telegram_bot.is_configured:
            await telegram_bot.dispatch_recommendations(recommendations, regime_result.regime.value)
            await telegram_bot.dispatch_ledger_update(update_summary)

        logger.info("17:00 cycle completed successfully for %s", target_date)
        return 0
    except DataUnavailableException as exc:
        logger.error("DATA_UNAVAILABLE: %s", exc)
        return 1
    except Exception:
        logger.exception("Automated cycle failed closed")
        return 1
    finally:
        await bulk_loader.close()
        await index_provider.close()


def setup_windows_task_scheduler() -> None:
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())
    task_name = "NSESwingAI_Daily_5PM_Scan"
    cmd = (
        f'schtasks /create /tn "{task_name}" /tr "\"{python_exe}\" \"{script_path}\" --run-now" '
        f'/sc daily /st 17:00 /f'
    )
    logger.info("Registering Windows Task Scheduler job at 17:00 IST")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        logger.info("Windows Task Scheduler registration succeeded")
    else:
        logger.error("Windows Task Scheduler registration failed: %s", res.stderr)


def run_continuous_loop() -> None:
    logger.info("Continuous scheduler active; waiting for 17:00 IST on trading days")
    last_run_date: date | None = None
    while True:
        now = datetime.now(IST)
        if now.hour == 17 and now.minute == 0 and last_run_date != now.date():
            asyncio.run(execute_daily_5pm_cycle(now.date()))
            last_run_date = now.date()
        time.sleep(30)


def main() -> None:
    args = parse_args()
    if args.setup_windows_task:
        setup_windows_task_scheduler()
        return
    if args.run_now or args.date:
        target = date.fromisoformat(args.date) if args.date else date.today()
        sys.exit(asyncio.run(execute_daily_5pm_cycle(target, force=args.force)))
    run_continuous_loop()


if __name__ == "__main__":
    main()
