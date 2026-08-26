#!/usr/bin/env python3
"""Automated daily signal cycle.

Production signal path:
  NSE universe/EOD + Upstox authenticated index/live data + IndianAPI intelligence
  -> technical screener -> CIO -> empirical probability/EV -> risk -> final basket.

The scheduler is deliberately fail-closed: if live market-regime inputs or the
empirical probability store are unavailable, it produces zero trade signals.
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from config.market_hours import MarketCalendar
from config.settings import settings
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.exceptions import DataUnavailableException
from src.core.models import SymbolMetadata
from src.data.historical_provider import HistoricalDataProvider
from src.data.indianapi_provider import IndianAPIProvider
from src.data.nse_provider import NseDataProvider
from src.data.upstox_provider import UpstoxDataProvider
from src.database.connection import get_db_session, init_db
from src.database.repository import DatabaseRepository
from src.quant.regime import MarketRegimeClassifier
from src.quant.screener import QuantScreener
from src.quant.probability_engine import HistoricalSetupOutcomeStore
from src.shadow.excel_ledger import TradeLedgerExcelManager
from src.shadow.telegram_bot import TelegramBotNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("run_automated_scheduler")


def parse_args():
    parser = argparse.ArgumentParser(description="NSE Swing AI production signal cycle")
    parser.add_argument("--run-now", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def execute_daily_5pm_cycle(target_date: date | None = None, force: bool = False) -> int:
    target_date = MarketCalendar.get_latest_trading_day(target_date or date.today())
    run_id = f"AUTO-5PM-{target_date:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    logger.info("NSE SWING AI production cycle | run=%s | date=%s", run_id, target_date)
    if not force and not MarketCalendar.is_trading_day(target_date):
        return 0

    init_db()
    nse_provider = NseDataProvider()
    hist_provider = HistoricalDataProvider()
    upstox = None
    indianapi = None
    telegram_bot = TelegramBotNotifier()
    ledger_manager = TradeLedgerExcelManager()

    try:
        if settings.UPSTOX_ENABLED and settings.UPSTOX_ACCESS_TOKEN:
            upstox = UpstoxDataProvider()
            universe_meta = await upstox.fetch_active_securities()
        else:
            universe_meta = await nse_provider.fetch_active_securities()
        if not universe_meta:
            raise DataUnavailableException("DATA_UNAVAILABLE: dynamic NSE universe is empty")

        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(target_date)
        if bhavcopy_df.empty:
            raise DataUnavailableException(f"DATA_UNAVAILABLE: NSE Bhavcopy unavailable for {target_date}")

        start_history_date = target_date - timedelta(days=420)
        stock_dfs = {}
        semaphore = asyncio.Semaphore(12)

        async def load_one(meta: SymbolMetadata):
            async with semaphore:
                try:
                    if upstox is not None:
                        df = await upstox.get_daily_ohlcv(meta.symbol, start_history_date, target_date)
                    else:
                        df = await hist_provider.get_daily_ohlcv(meta.symbol, start_history_date, target_date, min_bars=50)
                    return meta.symbol, df
                except Exception:
                    return meta.symbol, None

        loaded = await asyncio.gather(*(load_one(m) for m in universe_meta))
        stock_dfs = {symbol: df for symbol, df in loaded if df is not None and not df.empty}
        logger.info("Historical stock data loaded: %d/%d", len(stock_dfs), len(universe_meta))

        if upstox is None:
            raise DataUnavailableException("SIGNAL_MODE_REQUIRES_UPSTOX: configure UPSTOX_ACCESS_TOKEN for live NIFTY regime data")
        # Upstox instrument search/documentation uses NIFTY as the trading_symbol for
        # the NIFTY 50 index; resolve that canonical symbol rather than a display name.
        nifty_df = await upstox.get_index_daily_ohlcv("NIFTY", start_history_date, target_date)
        breadth = await nse_provider.get_market_breadth("NIFTY 500")

        breadth_population = 0
        above_50 = 0
        for df in stock_dfs.values():
            if len(df) < 50:
                continue
            breadth_population += 1
            above_50 += int(float(df["close"].iloc[-1]) > float(df["close"].tail(50).mean()))
        if breadth_population < 100:
            raise DataUnavailableException(f"Insufficient breadth population: {breadth_population}")
        pct_above_50_sma = above_50 / breadth_population * 100.0

        regime_result = MarketRegimeClassifier.classify_regime(nifty_df=nifty_df, advance_decline_ratio=breadth.advance_decline_ratio, pct_above_50_sma=pct_above_50_sma, india_vix=breadth.india_vix, as_of_date=target_date)
        logger.info("Regime=%s | stance=%s | A/D=%.2f | >50SMA=%.1f%% | VIX=%.2f", regime_result.regime.value, regime_result.trading_stance.value, breadth.advance_decline_ratio, pct_above_50_sma, breadth.india_vix or 0.0)
        if not regime_result.allow_long_swing_trades:
            logger.info("Regime veto: zero long signals")
            return 0

        HistoricalSetupOutcomeStore.load_from_disk()
        if not HistoricalSetupOutcomeStore._records:
            raise DataUnavailableException("SIGNAL_MODE_BLOCKED: empirical historical setup outcome store is empty")

        screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
        candidates = screener.screen_universe(universe_meta, stock_dfs, nifty_df)[:100]
        shared_context = {}
        if settings.INDIANAPI_ENABLED and settings.INDIANAPI_API_KEY:
            indianapi = IndianAPIProvider()
            shared_context["indianapi_provider"] = indianapi

        cio = CIOOrchestrator()
        recommendations = await cio.run_daily_scan(candidates=candidates, stock_dfs=stock_dfs, universe={m.symbol: m for m in universe_meta}, regime_result=regime_result, run_id=run_id, shared_context=shared_context)
        if recommendations:
            ledger_manager.record_recommendations(recommendations)
            with get_db_session() as session:
                repo = DatabaseRepository(session)
                repo.create_agent_run(run_id=run_id, market_regime=regime_result.regime.value, universe_size=len(universe_meta), quant_candidates_count=len(candidates))
                for rec in recommendations:
                    repo.save_trade_recommendation(rec)
                repo.complete_agent_run(run_id, "COMPLETED", len(recommendations))
            if telegram_bot.is_configured:
                await telegram_bot.dispatch_recommendations(recommendations, regime_result.regime.value)
        else:
            logger.info("NO TRADE: no candidate survived the complete CIO/risk/EV pipeline")

        update_summary = ledger_manager.update_ledger_with_eod_data(bhavcopy_df, target_date)
        if telegram_bot.is_configured:
            await telegram_bot.dispatch_ledger_update(update_summary)
        return 0
    except DataUnavailableException as exc:
        logger.error("SIGNAL CYCLE BLOCKED: %s", exc)
        return 1
    except Exception:
        logger.exception("Production signal cycle failed")
        return 1
    finally:
        await nse_provider.close()
        await hist_provider.nse_provider.close()
        if upstox is not None:
            await upstox.close()
        if indianapi is not None:
            await indianapi.close()


def main():
    args = parse_args()
    if args.run_now or args.date:
        target = date.fromisoformat(args.date) if args.date else date.today()
        raise SystemExit(asyncio.run(execute_daily_5pm_cycle(target, force=args.force)))
    while True:
        now = datetime.now()
        if now.hour == 17 and now.minute == 0:
            asyncio.run(execute_daily_5pm_cycle(now.date()))
            time.sleep(90)
        time.sleep(30)


if __name__ == "__main__":
    main()
