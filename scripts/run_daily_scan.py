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

    from src.data.historical_universe import HistoricalUniverseProvider
    from src.data.universe import UniverseDiscoveryEngine
    from src.candidate_discovery import CandidateDiscoveryEngine, CandidateDiscoveryConfig

    nse_provider = NseDataProvider()
    index_provider = NseIndexDataProvider()
    bulk_loader = BulkHistoricalLoader(nse_provider)
    try:
        universe_meta = await nse_provider.fetch_active_securities()
        if not universe_meta:
            active_base = UniverseDiscoveryEngine.get_default_active_universe()
            universe_meta = active_base
        
        # Filter universe by as_of_date for PIT integrity
        eligible_symbols = HistoricalUniverseProvider.get_universe_for_date(scan_date, securities=universe_meta)
        universe_meta = [m for m in universe_meta if m.symbol in eligible_symbols]
        logger.info("Official NSE equity universe: %d symbols", len(universe_meta))

        bhavcopy_df = await nse_provider.fetch_bhavcopy_for_date(scan_date)
        if bhavcopy_df.empty:
            raise DataUnavailableException(f"Empty NSE Bhavcopy for {scan_date}")

        # Same-day liquidity gate determines which symbols enter the expensive
        # specialist pipeline, but breadth is calculated from the ENTIRE NSE
        # universe, not from the filtered candidate set.
        bhavcopy_df["_symbol"] = bhavcopy_df["symbol"].astype(str).str.strip().str.upper()
        bhav_by_symbol = bhavcopy_df.set_index("_symbol")
        bhav_symbols = set(bhavcopy_df["_symbol"])
        filtered_meta = []
        for meta in universe_meta:
            if meta.symbol.upper() not in bhav_symbols:
                continue
            row = bhav_by_symbol.loc[meta.symbol.upper()]
            if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            close = float(row.get("close", 0.0))
            turnover = float(row.get("turnover_crores", 0.0))
            if close >= 20.0 and turnover >= 3.0:
                filtered_meta.append(meta)
        logger.info("Same-day liquidity gate: %d/%d symbols retained", len(filtered_meta), len(universe_meta))

        # One official Bhavcopy request per trading day for the entire NSE
        # universe. This preserves whole-market breadth while avoiding the
        # previous N-symbol x M-day request storm.
        start_history_date = scan_date - timedelta(days=400)
        stock_dfs = await bulk_loader.load(
            [m.symbol for m in universe_meta],
            start_history_date,
            scan_date,
            min_bars=100,
        )
        eligible_meta = [m for m in filtered_meta if m.symbol in stock_dfs]
        logger.info("Historical PIT data available for %d/%d NSE symbols", len(stock_dfs), len(universe_meta))

        nifty_df = await index_provider.get_index_history("NIFTY 50", start_history_date, scan_date)
        vix_df = await index_provider.get_india_vix_history(start_history_date, scan_date)
        if len(nifty_df) < 200:
            raise DataUnavailableException(f"NIFTY 50 history insufficient: {len(nifty_df)} bars < 200")

        # Market breadth is intentionally computed over all symbols with valid
        # historical observations, not just the tradeable shortlist.
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

        # Stage-1 Candidate Discovery Engine integration
        candidate_dfs = {m.symbol: stock_dfs[m.symbol] for m in eligible_meta if m.symbol in stock_dfs}
        discovery_results = CandidateDiscoveryEngine.discover_candidates(
            universe=eligible_meta,
            as_of_date=scan_date,
            market_data_map=candidate_dfs,
            config=CandidateDiscoveryConfig(min_price=20.0, min_average_turnover_crores=3.0, min_history_length=50),
        )
        discovered_eligible_meta = [
            m for m, res in zip(eligible_meta, discovery_results) if res.eligible
        ]

        screener = QuantScreener(min_adtv_crores=5.0, min_price=20.0)
        candidates = screener.screen_universe(discovered_eligible_meta, candidate_dfs, nifty_df)
        logger.info("Final Stage-1 candidates: %d", len(candidates))
        if not candidates:
            logger.info("NO TRADE TODAY: Stage-1 screener produced no candidates")
            return 0

        cio = CIOOrchestrator()
        universe = {m.symbol: m for m in eligible_meta}
        recommendations = await cio.run_daily_scan(
            candidates=candidates,
            stock_dfs=candidate_dfs,
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
<<<<<<< HEAD
    except DataUnavailableException as exc:
        logger.error("DATA_UNAVAILABLE: %s", exc)
        return 1
    except Exception:
        logger.exception("Daily scan failed closed due to unexpected error")
        return 1
    finally:
        await bulk_loader.close()
        await index_provider.close()
=======

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

    # 6. Fetch Multi-day Historical OHLCV for Pre-Filtered Candidates & Pass to CandidateDiscoveryEngine
    from src.candidate_discovery import CandidateDiscoveryConfig, CandidateDiscoveryEngine

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

    discovery_results = CandidateDiscoveryEngine.discover_candidates(
        universe=eligible_meta,
        as_of_date=scan_date,
        market_data_map=stock_dfs,
        config=CandidateDiscoveryConfig(min_price=20.0, min_average_turnover_crores=3.0, min_history_length=50),
    )

    discovered_eligible_meta = [
        m for m, res in zip(eligible_meta, discovery_results) if res.eligible
    ]

    candidates = screener.screen_universe(discovered_eligible_meta, stock_dfs, nifty_df)
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
>>>>>>> 66f33b9 (Real Trade Analysis & Scanner Integration: Fix universe initialization in run_daily_scan.py, integrate CandidateDiscoveryEngine, connect run_shadow_monitor.py to SQLite database, implement RealTradeAnalyzer in src/shadow/analyzer.py, CLI tool in scripts/analyze_trades.py, and test suite in tests/test_real_trade_analysis.py)


def main() -> None:
    args = parse_args()
    scan_date = date.fromisoformat(args.date) if args.date else get_latest_trading_day(date.today())
    sys.exit(asyncio.run(run_scan(scan_date, dry_run=args.dry_run, force=args.force)))


if __name__ == "__main__":
    main()
