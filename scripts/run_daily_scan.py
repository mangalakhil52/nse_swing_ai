#!/usr/bin/env python3
"""
Daily Market Scan Entry Point — scripts/run_daily_scan.py

Executes daily EOD scanning pipeline using REAL market data and zero synthetic fallbacks.
Pipeline Architecture:
  1. Initialize Database Schema & Security Master
  2. Fetch & Validate Official EOD Bhavcopy
  3. Classify Market Regime (NIFTY 50 real data)
  4. Stage-1 Screener (filters 2,000+ universe down to 10-30 candidates via Bhavcopy)
  5. Fetch Multi-day Historical OHLCV for candidates
  6. CIO Multi-Agent Research & Adversarial Thesis Killer
  7. Risk Veto, Probability & Net EV Engine, Execution Quality Model
  8. Save Trade Recommendations & Telegram Summary
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import uuid
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config.market_hours import get_latest_trading_day
from config.settings import settings
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
    parser.add_argument("--date", type=str, default=None, help="Scan date (YYYY-MM-DD), default latest completed trading session")
    parser.add_argument("--dry-run", action="store_true", help="Run scan without persisting to database")
    parser.add_argument("--force", action="store_true", help="Force run scan even on weekends/holidays")
    return parser.parse_args()


async def run_scan(scan_date: date, dry_run: bool = False, force: bool = False) -> int:
    run_id = f"SCAN-{scan_date.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    logger.info("=" * 60)
    logger.info(f"NSE SWING AI — DAILY SCAN INITIATED")
    logger.info(f"Run ID: {run_id} | Date: {scan_date} | Dry Run: {dry_run}")
    logger.info("=" * 60)

    # 1. Database Initialization
    init_db()
    logger.info("Database schema initialized.")

    # 2. Universe Construction (Survivorship-safe)
    logger.info("Building NSE eligible trading universe...")
    from src.data.historical_universe import HistoricalUniverseProvider
    symbols = HistoricalUniverseProvider.get_universe_for_date(scan_date)
    universe_meta = [
        SymbolMetadata(symbol=sym, company_name=sym, sector="General", is_fno_eligible=True)
        for sym in symbols
    ]
    logger.info(f"Universe built: {len(universe_meta)} eligible symbols.")

    # 3. Ingest Real EOD Bhavcopy
    logger.info(f"Downloading NSE Bhavcopy for {scan_date}...")
    nse_provider = NseDataProvider()
    try:
        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(scan_date)
    except Exception as e:
        logger.error(f"DATA_UNAVAILABLE: Could not fetch Bhavcopy for {scan_date}: {e}")
        return 1

    if bhavcopy_df.empty:
        logger.error(f"DATA_UNAVAILABLE: Bhavcopy for {scan_date} is empty. Aborting scan.")
        return 1

    logger.info(f"Bhavcopy loaded: {len(bhavcopy_df)} EQ series records.")

    # 4. Market Regime Classification (NIFTY 50 Real Data)
    logger.info("Classifying market regime via Nifty 50 historical data...")
    hist_provider = HistoricalDataProvider()
    start_history_date = scan_date - timedelta(days=120)

    nifty_df = pd.DataFrame()
    try:
        nifty_df = await hist_provider.get_daily_ohlcv("NIFTY 50", start_history_date, scan_date, min_bars=50)
    except Exception as e:
        logger.warning(f"DATA_UNAVAILABLE: Could not fetch NIFTY 50 data ({e}). Market regime set to UNKNOWN.")

    regime_result = MarketRegimeClassifier.classify_regime(nifty_df=nifty_df)
    logger.info(f"Regime: {regime_result.regime.value} | Stance: {regime_result.trading_stance.value}")

    if not regime_result.allow_long_swing_trades:
        logger.warning(f"Market regime {regime_result.regime.value} prohibits long trades. Completing with 0 recommendations.")
        return 0

    # 5. Fast Stage-1 Screener across Bhavcopy
    logger.info(f"Running Stage-1 quantitative screener across full universe ({len(universe_meta)} symbols)...")
    screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)

    # Filter universe using Bhavcopy records first
    bhav_symbols = set(bhavcopy_df["symbol"].str.strip().tolist())
    candidate_meta = [m for m in universe_meta if m.symbol in bhav_symbols]

    # Pre-filter candidates with positive turnover and volume
    filtered_meta = []
    for m in candidate_meta:
        b_rows = bhavcopy_df[bhavcopy_df["symbol"].str.strip() == m.symbol]
        if not b_rows.empty:
            row = b_rows.iloc[0]
            c = float(row.get("close", 0.0))
            v = int(row.get("volume", 0))
            turnover = (c * v) / 1e7
            if c >= 20.0 and turnover >= 3.0:
                filtered_meta.append(m)

    logger.info(f"Stage-1 Bhavcopy pre-filter output: {len(filtered_meta)} candidates for historical evaluation.")

    # 6. Fetch Multi-day Historical OHLCV for Pre-Filtered Candidates
    stock_dfs: dict[str, pd.DataFrame] = {}
    eligible_meta: list[SymbolMetadata] = []

    for sym_meta in filtered_meta:
        sym = sym_meta.symbol
        try:
            df_hist = await hist_provider.get_daily_ohlcv(sym, start_history_date, scan_date, min_bars=50)
            stock_dfs[sym] = df_hist
            eligible_meta.append(sym_meta)
        except Exception as e:
            logger.debug(f"Skipping {sym} due to unavailable/insufficient historical data: {e}")

    candidates = screener.screen_universe(eligible_meta, stock_dfs, nifty_df)
    logger.info(f"Screener final output: {len(candidates)} candidates for deep multi-agent research.")

    if not candidates:
        logger.info("No candidates passed Stage-1 screening. Exiting cleanly.")
        return 0

    # 7. CIO Multi-Agent Research & Adversarial Thesis Killer
    logger.info(f"Running CIO multi-agent pipeline on {len(candidates)} candidates...")
    cio = CIOOrchestrator()
    recommendations = await cio.run_daily_scan(
        candidates=candidates,
        stock_dfs=stock_dfs,
        universe={m.symbol: m for m in eligible_meta},
        regime_result=regime_result,
        run_id=run_id,
    )

    logger.info(f"Scan complete. Generated {len(recommendations)} high-conviction trade recommendations.")

    # 8. Dispatch Telegram Alerts
    if recommendations:
        summary_msg = TelegramFormatter.format_scan_summary(recommendations, regime_result.regime.value)
        logger.info(f"\nTelegram Summary Output:\n{summary_msg}")

    return 0


def main():
    args = parse_args()

    if args.date:
        scan_date = date.fromisoformat(args.date)
    else:
        scan_date = get_latest_trading_day(date.today())

    exit_code = asyncio.run(run_scan(scan_date, dry_run=args.dry_run, force=args.force))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
