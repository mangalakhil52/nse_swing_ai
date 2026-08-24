#!/usr/bin/env python3
"""Daily EOD scan using the complete official NSE equity universe."""

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
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.models import SymbolMetadata
from src.data.historical_provider import HistoricalDataProvider
from src.data.nse_provider import NseDataProvider
from src.database.connection import init_db
from src.quant.regime import MarketRegimeClassifier
from src.quant.screener import QuantScreener
from src.shadow.alerts import TelegramFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily_scan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI Daily Scanner")
    parser.add_argument("--date", type=str, default=None, help="Scan date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run without persisting")
    parser.add_argument("--force", action="store_true", help="Force run on non-trading days")
    return parser.parse_args()


async def run_scan(scan_date: date, dry_run: bool = False, force: bool = False) -> int:
    run_id = f"SCAN-{scan_date:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    logger.info("NSE SWING AI scan started | run=%s | date=%s", run_id, scan_date)
    init_db()

    nse_provider = NseDataProvider()
    try:
        # The only live-universe source is NSE's official EQUITY_L security master.
        universe_meta = await nse_provider.fetch_active_securities()
        if not universe_meta:
            logger.error("DATA_UNAVAILABLE: NSE equity master unavailable; refusing partial-universe scan")
            return 1
        logger.info("Official NSE equity universe: %d symbols", len(universe_meta))

        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(scan_date)
        if bhavcopy_df.empty:
            logger.error("DATA_UNAVAILABLE: empty NSE Bhavcopy for %s", scan_date)
            return 1
        logger.info("Bhavcopy loaded: %d records", len(bhavcopy_df))

        hist_provider = HistoricalDataProvider()
        start_history_date = scan_date - timedelta(days=120)
        nifty_df = pd.DataFrame()
        try:
            nifty_df = await hist_provider.get_daily_ohlcv("NIFTY 50", start_history_date, scan_date, min_bars=50)
        except Exception as exc:
            logger.warning("NIFTY 50 data unavailable: %s", exc)

        regime_result = MarketRegimeClassifier.classify_regime(nifty_df=nifty_df)
        logger.info("Regime=%s | stance=%s", regime_result.regime.value, regime_result.trading_stance.value)
        if not regime_result.allow_long_swing_trades:
            logger.warning("Regime prohibits long swing trades; returning zero recommendations")
            return 0

        screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
        bhavcopy_df["_symbol"] = bhavcopy_df["symbol"].astype(str).str.strip().str.upper()
        bhav_by_symbol = bhavcopy_df.set_index("_symbol")
        bhav_symbols = set(bhavcopy_df["_symbol"])
        candidate_meta = [m for m in universe_meta if m.symbol.upper() in bhav_symbols]

        # Cheap EOD pre-filter only; full historical screener remains authoritative.
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

        stock_dfs: dict[str, pd.DataFrame] = {}
        eligible_meta: list[SymbolMetadata] = []
        for meta in filtered_meta:
            try:
                df_hist = await hist_provider.get_daily_ohlcv(meta.symbol, start_history_date, scan_date, min_bars=50)
                if df_hist is not None and not df_hist.empty:
                    stock_dfs[meta.symbol] = df_hist
                    eligible_meta.append(meta)
            except Exception as exc:
                logger.debug("Skipping %s: %s", meta.symbol, exc)

        candidates = screener.screen_universe(eligible_meta, stock_dfs, nifty_df)
        logger.info("Final Stage-1 candidates: %d", len(candidates))
        if not candidates:
            return 0

        cio = CIOOrchestrator()
        recommendations = await cio.run_daily_scan(
            candidates=candidates,
            stock_dfs=stock_dfs,
            universe={m.symbol: m for m in eligible_meta},
            regime_result=regime_result,
            run_id=run_id,
        )
        logger.info("Scan complete: %d recommendations", len(recommendations))

        if recommendations:
            summary = TelegramFormatter.format_scan_summary(recommendations, regime_result.regime.value)
            logger.info("Telegram Summary Output:\n%s", summary)
        return 0
    finally:
        await nse_provider.close()


def main() -> None:
    args = parse_args()
    scan_date = date.fromisoformat(args.date) if args.date else get_latest_trading_day(date.today())
    sys.exit(asyncio.run(run_scan(scan_date, dry_run=args.dry_run, force=args.force)))


if __name__ == "__main__":
    main()
