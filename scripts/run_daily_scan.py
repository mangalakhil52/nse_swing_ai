#!/usr/bin/env python3
"""Daily EOD NSE swing scan with strict PIT and data-quality controls."""

import argparse
import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_hours import get_latest_trading_day, is_trading_day
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.exceptions import DataUnavailableException
from src.data.bulk_history import BulkHistoricalLoader
from src.data.nse_index_provider import NseIndexDataProvider
from src.data.nse_provider import NseDataProvider
from src.database.connection import init_db
from src.quant.regime import MarketRegimeClassifier
from src.quant.regime_inputs import compute_breadth, latest_vix
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
    parser.add_argument("--dry-run", action="store_true", help="Run without persistence")
    parser.add_argument("--force", action="store_true", help="Allow execution when the supplied date is not a trading day")
    return parser.parse_args()


async def run_scan(scan_date: date, dry_run: bool = False, force: bool = False) -> int:
    run_id = f"SCAN-{scan_date:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    logger.info("NSE SWING AI scan started | run=%s | date=%s", run_id, scan_date)
    init_db()

    if not force and not is_trading_day(scan_date):
        logger.info("%s is not an NSE trading day; no scan executed.", scan_date)
        return 0

    nse_provider = NseDataProvider()
    index_provider = NseIndexDataProvider()
    bulk_loader = BulkHistoricalLoader(nse_provider)
    try:
        universe_meta = await nse_provider.fetch_active_securities()
        if not universe_meta:
            raise DataUnavailableException("NSE equity master unavailable; refusing partial-universe scan")
        logger.info("Official NSE equity universe: %d symbols", len(universe_meta))

        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(scan_date)
        if bhavcopy_df.empty:
            raise DataUnavailableException(f"Empty NSE Bhavcopy for {scan_date}")

        # Cheap same-day gate before historical loading.
        bhavcopy_df["_symbol"] = bhavcopy_df["symbol"].astype(str).str.strip().str.upper()
        bhav_by_symbol = bhavcopy_df.set_index("_symbol")
        bhav_symbols = set(bhavcopy_df["_symbol"])
        candidate_meta = [m for m in universe_meta if m.symbol.upper() in bhav_symbols]
        filtered_meta = []
        for meta in candidate_meta:
            row = bhav_by_symbol.loc[meta.symbol.upper()]
            if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            close = float(row.get("close", 0.0))
            turnover = float(row.get("turnover_crores", 0.0))
            if close >= 20.0 and turnover >= 3.0:
                filtered_meta.append(meta)
        logger.info("Same-day liquidity gate: %d/%d symbols retained", len(filtered_meta), len(candidate_meta))

        # 400 calendar days gives enough observations for 52W structure and long EMAs.
        start_history_date = scan_date - timedelta(days=400)
        stock_dfs = await bulk_loader.load(
            [m.symbol for m in filtered_meta],
            start_history_date,
            scan_date,
            min_bars=100,
        )
        eligible_meta = [m for m in filtered_meta if m.symbol in stock_dfs]
        logger.info("Historical PIT data available for %d symbols", len(eligible_meta))

        nifty_df = await index_provider.get_index_history("NIFTY 50", start_history_date, scan_date)
        vix_df = await index_provider.get_india_vix_history(start_history_date, scan_date)
        if len(nifty_df) < 200:
            raise DataUnavailableException(f"NIFTY 50 history insufficient: {len(nifty_df)} bars < 200")

        ad_ratio, pct_above_50 = compute_breadth(stock_dfs, scan_date)
        vix = latest_vix(vix_df, scan_date)
        regime_result = MarketRegimeClassifier.classify_regime(
            nifty_df=nifty_df,
            advance_decline_ratio=ad_ratio,
            pct_above_50_sma=pct_above_50,
            india_vix=vix,
            as_of_date=scan_date,
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
            logger.warning("Market regime blocks long swing trades; returning NO TRADE TODAY")
            return 0

        screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
        candidates = screener.screen_universe(eligible_meta, stock_dfs, nifty_df)
        logger.info("Final Stage-1 candidates: %d", len(candidates))
        if not candidates:
            logger.info("NO TRADE TODAY: Stage-1 screener produced no candidates")
            return 0

        cio = CIOOrchestrator()
        universe = {m.symbol: m for m in eligible_meta}
        recommendations = await cio.run_daily_scan(
            candidates=candidates,
            stock_dfs=stock_dfs,
            universe=universe,
            regime_result=regime_result,
            run_id=run_id,
        )
        logger.info("Scan complete: %d recommendations", len(recommendations))

        if recommendations:
            summary = TelegramFormatter.format_scan_summary(recommendations, regime_result.regime.value)
            logger.info("Telegram Summary Output:\n%s", summary)
        else:
            logger.info("NO TRADE TODAY: CIO rejected all candidates")
        return 0
    except DataUnavailableException as exc:
        logger.error("DATA_UNAVAILABLE: %s", exc)
        return 1
    except Exception:
        logger.exception("Daily scan failed closed due to unexpected error")
        return 1
    finally:
        await bulk_loader.close()
        await index_provider.close()


def main() -> None:
    args = parse_args()
    scan_date = date.fromisoformat(args.date) if args.date else get_latest_trading_day(date.today())
    sys.exit(asyncio.run(run_scan(scan_date, dry_run=args.dry_run, force=args.force)))


if __name__ == "__main__":
    main()
