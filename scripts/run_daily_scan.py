#!/usr/bin/env python3
"""Daily EOD scan using the complete official NSE equity universe.

Pipeline:
  dynamic universe -> EOD exchange snapshot -> liquidity gate -> historical data
  -> complete market regime -> technical screen -> top 100 -> CIO research
  -> empirical EV/risk gates -> top 0-3 recommendations.

No hardcoded stock list and no synthetic market-data fallback are permitted.
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import uuid
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config.market_hours import get_latest_trading_day
from config.settings import settings
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.exceptions import DataUnavailableException
from src.core.models import SymbolMetadata
from src.data.historical_provider import HistoricalDataProvider
from src.data.nse_provider import NseDataProvider
from src.data.upstox_provider import UpstoxDataProvider
from src.database.connection import init_db
from src.quant.regime import MarketRegimeClassifier
from src.quant.screener import QuantScreener
from src.shadow.alerts import TelegramFormatter
from web.api_contract import BUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("run_daily_scan")


async def _load_history(symbol: str, start: date, end: date, upstox: UpstoxDataProvider | None, hist: HistoricalDataProvider) -> pd.DataFrame:
    if upstox is not None:
        return await upstox.get_daily_ohlcv(symbol, start, end)
    return await hist.get_daily_ohlcv(symbol, start, end, min_bars=50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Daily Scanner")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def run_scan(scan_date: date, dry_run: bool = False, force: bool = False) -> int:
    run_id = f"SCAN-{scan_date:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    logger.info("NSE SWING AI scan started | run=%s | date=%s", run_id, scan_date)
    BUS.publish({"type": "connection", "connected": True})
    BUS.publish({"type": "scan_progress", "universe": 0, "filtered": 0, "candidates": 0, "intel": 0, "final": 0, "processed": 0, "status": "STARTING"})
    init_db()

    nse_provider = NseDataProvider()
    upstox: UpstoxDataProvider | None = None
    hist_provider: HistoricalDataProvider | None = None
    try:
        if settings.UPSTOX_ENABLED and settings.UPSTOX_ACCESS_TOKEN:
            upstox = UpstoxDataProvider()
            universe_meta = await upstox.fetch_active_securities()
            logger.info("Upstox dynamic NSE equity universe: %d symbols", len(universe_meta))
        else:
            universe_meta = await nse_provider.fetch_active_securities()
            logger.info("Official NSE equity universe: %d symbols", len(universe_meta))
        if not universe_meta:
            raise DataUnavailableException("DATA_UNAVAILABLE: dynamic NSE equity master is empty")
        BUS.publish({"type": "scan_progress", "universe": len(universe_meta), "status": "UNIVERSE_LOADED"})

        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(scan_date)
        if bhavcopy_df.empty:
            raise DataUnavailableException(f"DATA_UNAVAILABLE: empty NSE Bhavcopy for {scan_date}")
        logger.info("Bhavcopy loaded: %d records", len(bhavcopy_df))

        hist_provider = HistoricalDataProvider()
        start_history_date = scan_date - timedelta(days=420)
        screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
        bhavcopy_df["_symbol"] = bhavcopy_df["symbol"].astype(str).str.strip().str.upper()
        bhav_by_symbol = bhavcopy_df.set_index("_symbol")
        bhav_symbols = set(bhavcopy_df["_symbol"])
        candidate_meta = [m for m in universe_meta if m.symbol.upper() in bhav_symbols]

        filtered_meta: list[SymbolMetadata] = []
        for meta in candidate_meta:
            row = bhav_by_symbol.loc[meta.symbol.upper()]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            close = float(row.get("close", 0.0))
            volume = float(row.get("volume", 0.0))
            turnover = float(row.get("turnover_crores", (close * volume) / 1e7))
            if close >= 20.0 and turnover >= 3.0:
                filtered_meta.append(meta)
        logger.info("Bhavcopy pre-filter: %d candidates", len(filtered_meta))
        BUS.publish({"type": "scan_progress", "filtered": len(filtered_meta), "status": "LIQUIDITY_FILTER"})

        stock_dfs: dict[str, pd.DataFrame] = {}
        eligible_meta: list[SymbolMetadata] = []
        semaphore = asyncio.Semaphore(12)

        async def fetch_one(meta: SymbolMetadata):
            async with semaphore:
                try:
                    df_hist = await _load_history(meta.symbol, start_history_date, scan_date, upstox, hist_provider)
                    if df_hist is not None and not df_hist.empty:
                        return meta, df_hist
                except Exception as exc:
                    logger.debug("Skipping %s: %s", meta.symbol, exc)
                return meta, None

        results = await asyncio.gather(*(fetch_one(meta) for meta in filtered_meta))
        for meta, df_hist in results:
            if df_hist is not None:
                stock_dfs[meta.symbol] = df_hist
                eligible_meta.append(meta)
        logger.info("Historical data loaded: %d/%d symbols", len(eligible_meta), len(filtered_meta))
        BUS.publish({"type": "scan_progress", "processed": len(eligible_meta), "status": "HISTORY_LOADED"})

        # Complete market-regime inputs are now calculated from genuine observations.
        # A/D and India VIX come from NSE live market data; breadth participation is
        # calculated from the same historical stock universe already loaded above.
        breadth = await nse_provider.get_market_breadth("NIFTY 500")
        nifty_df = pd.DataFrame()
        try:
            if upstox is not None:
                nifty_df = await upstox.get_index_daily_ohlcv("NIFTY 50", start_history_date, scan_date)
            else:
                nifty_df = await hist_provider.get_daily_ohlcv("NIFTY 50", start_history_date, scan_date, min_bars=50)
        except Exception as exc:
            raise DataUnavailableException(f"NIFTY 50 history unavailable: {exc}") from exc

        above_50 = 0
        breadth_population = 0
        for df in stock_dfs.values():
            if df is None or len(df) < 50:
                continue
            close = float(df["close"].iloc[-1])
            sma50 = float(df["close"].tail(50).mean())
            breadth_population += 1
            above_50 += int(close > sma50)
        if breadth_population < 100:
            raise DataUnavailableException(f"Insufficient breadth population: {breadth_population} stocks")
        pct_above_50_sma = above_50 / breadth_population * 100.0

        regime_result = MarketRegimeClassifier.classify_regime(
            nifty_df=nifty_df,
            advance_decline_ratio=breadth.advance_decline_ratio,
            pct_above_50_sma=pct_above_50_sma,
            india_vix=breadth.india_vix,
            as_of_date=scan_date,
        )
        logger.info("Regime=%s | stance=%s | A/D=%.2f | >50SMA=%.1f%% | VIX=%.2f", regime_result.regime.value, regime_result.trading_stance.value, breadth.advance_decline_ratio, pct_above_50_sma, breadth.india_vix or 0.0)
        BUS.publish({"type": "alert", "severity": "green", "message": f"Regime {regime_result.regime.value} | A/D {breadth.advance_decline_ratio:.2f} | >50SMA {pct_above_50_sma:.1f}% | VIX {breadth.india_vix:.2f}"})
        if not regime_result.allow_long_swing_trades:
            BUS.publish({"type": "scan_progress", "status": "NO_TRADE_REGIME"})
            return 0

        candidates = screener.screen_universe(eligible_meta, stock_dfs, nifty_df)[:100]
        logger.info("Stage-1 candidates passed to CIO: %d", len(candidates))
        BUS.publish({"type": "scan_progress", "candidates": len(candidates), "intel": len(candidates), "status": "TECHNICAL_SCREEN_COMPLETE"})
        if not candidates:
            BUS.publish({"type": "scan_progress", "final": 0, "status": "NO_TRADE"})
            return 0

        cio = CIOOrchestrator()
        recommendations = await cio.run_daily_scan(candidates=candidates, stock_dfs=stock_dfs,
            universe={m.symbol: m for m in eligible_meta}, regime_result=regime_result, run_id=run_id)
        logger.info("Scan complete: %d recommendations", len(recommendations))
        BUS.publish({"type": "scan_progress", "final": len(recommendations), "status": "COMPLETE" if recommendations else "NO_TRADE"})

        if recommendations:
            for rec in recommendations:
                BUS.publish({"type": "agent", "agent": "CIO FUSION", "status": "PASS", "progress": 100,
                    "processed": len(candidates), "decision": getattr(rec, "conviction_grade", "TRADE"),
                    "log": [f"{rec.symbol}: final recommendation", f"Entry {rec.trade_levels.entry_trigger_price:.2f}", f"T1 {rec.trade_levels.target_1:.2f}", f"SL {rec.trade_levels.stop_loss_price:.2f}"]})
            logger.info("Telegram Summary Output:\n%s", TelegramFormatter.format_scan_summary(recommendations, regime_result.regime.value))
        return 0
    except DataUnavailableException as exc:
        logger.error("%s", exc)
        BUS.publish({"type": "alert", "severity": "red", "message": str(exc)})
        BUS.publish({"type": "scan_progress", "status": "DATA_UNAVAILABLE"})
        return 1
    except Exception as exc:
        logger.exception("Scan failed")
        BUS.publish({"type": "alert", "severity": "red", "message": f"Backend failure: {exc}"})
        BUS.publish({"type": "scan_progress", "status": "FAILED"})
        return 1
    finally:
        await nse_provider.close()
        if upstox is not None:
            await upstox.close()
        BUS.publish({"type": "connection", "connected": False})


def main() -> None:
    args = parse_args()
    scan_date = date.fromisoformat(args.date) if args.date else get_latest_trading_day(date.today())
    raise SystemExit(asyncio.run(run_scan(scan_date, dry_run=args.dry_run, force=args.force)))


if __name__ == "__main__":
    main()
