#!/usr/bin/env python3
"""
Pre-Market 8:00 AM News & Catalyst Scan — scripts/run_premarket_news_scan.py

Runs automatically every trading day at 08:00 AM IST (before NSE pre-open at 09:00 AM):
  1. Checks if today is an official NSE trading day.
  2. Fetches overnight corporate announcements, board meeting filings, order wins, and news.
  3. Evaluates active portfolio stocks & watchlist for key morning catalysts.
  4. Dispatches a Pre-Market News Bulletin alert to Telegram.
  5. Logs all news evidence to the database.

Usage:
  python scripts/run_premarket_news_scan.py [--run-now] [--force]
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_hours import MarketCalendar
from src.agents.catalyst_agent import CatalystAgent
from src.agents.news_agent import NewsIntelligenceAgent
from src.core.evidence import EvidenceGraph
from src.core.models import SymbolMetadata
from src.data.news_provider import FinancialNewsProvider
from src.data.nse_provider import NseDataProvider
from src.data.universe import UniverseDiscoveryEngine
from src.database.connection import init_db
from src.shadow.telegram_bot import TelegramBotNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_premarket_news_scan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE Swing AI 8:00 AM Pre-Market News Scan")
    parser.add_argument("--run-now", action="store_true", help="Execute pre-market scan immediately")
    parser.add_argument("--force", action="store_true", help="Force scan execution even on weekends/holidays")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    return parser.parse_args()


async def execute_premarket_news_scan(target_date: date | None = None, force: bool = False) -> int:
    """Executes the 8:00 AM Pre-Market News & Catalyst Intelligence scan."""
    target_date = target_date or date.today()
    run_id = f"NEWS-8AM-{target_date.strftime('%Y%m%d')}-{int(datetime.now().timestamp())}"

    logger.info(f"{'='*60}")
    logger.info(f"NSE SWING AI — 8:00 AM PRE-MARKET NEWS & CATALYST SCAN")
    logger.info(f"Run ID: {run_id} | Date: {target_date} | Force: {force}")
    logger.info(f"{'='*60}")

    if not force and not MarketCalendar.is_trading_day(target_date):
        logger.info(f"{target_date} is a weekend or NSE market holiday. Skipping pre-market scan. Use --force to override.")
        return 0

    init_db()
    news_provider = FinancialNewsProvider()
    telegram_bot = TelegramBotNotifier()
    news_agent = NewsIntelligenceAgent()
    catalyst_agent = CatalystAgent()

    # Load active universe
    nse_provider = NseDataProvider()
    universe_engine = UniverseDiscoveryEngine(market_data_provider=nse_provider)
    universe_meta = await universe_engine.build_universe()
    await nse_provider.close()

    evidence_graph = EvidenceGraph(run_id=run_id)
    catalyst_alerts = []
    news_bulletins = []

    # Scan universe for overnight filings & news
    logger.info(f"Scanning overnight corporate announcements across {len(universe_meta)} symbols...")

    for meta in universe_meta[:60]:  # Priority focus on top liquid symbols
        sym = meta.symbol
        try:
            announcements = await news_provider.fetch_company_announcements(sym, lookback_days=2)
            articles = await news_provider.fetch_latest_news(sym, lookback_days=2)

            if not announcements and not articles:
                continue

            ctx = {"announcements": announcements, "news_articles": articles}

            # Run News & Catalyst Agents
            import pandas as pd
            df_dummy = pd.DataFrame()
            news_out = await news_agent.execute(meta, df_dummy, evidence_graph, run_id, ctx)
            catalyst_out = await catalyst_agent.execute(meta, df_dummy, evidence_graph, run_id, ctx)

            if catalyst_out.score >= 70.0 or news_out.signal.value == "BULLISH":
                desc = catalyst_out.metrics.get("description", "Positive Corporate Filing")
                catalyst_alerts.append(
                    f"⚡ *{sym}* ({meta.sector}) | Score: *{catalyst_out.score:.0f}/100*\n"
                    f"   • {desc}\n"
                    f"   • Net Sentiment: {news_out.metrics.get('sentiment', 'POSITIVE')}"
                )

            elif news_out.signal.value == "BEARISH":
                news_bulletins.append(
                    f"⚠️ *{sym}* — Negative News/Filing Detected\n"
                    f"   • {news_out.risks_identified[0] if news_out.risks_identified else 'Negative sentiment'}"
                )

        except Exception as e:
            logger.debug(f"Error checking news for {sym}: {e}")

    # Build Pre-Market Telegram Bulletin
    msg_lines = [
        f"🌅 *NSE SWING AI — 8:00 AM PRE-MARKET BULLETIN*",
        f"📅 Date: {target_date.strftime('%Y-%m-%d')} | Session: *Pre-Market*",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if catalyst_alerts:
        msg_lines.append(f"*🔥 KEY OVERNIGHT CATALYSTS & FILINGS ({len(catalyst_alerts)}):*")
        msg_lines.extend(catalyst_alerts[:5])
        msg_lines.append("")

    if news_bulletins:
        msg_lines.append(f"*⚠️ NEGATIVE NEWS / RISK WARNINGS ({len(news_bulletins)}):*")
        msg_lines.extend(news_bulletins[:3])
        msg_lines.append("")

    if not catalyst_alerts and not news_bulletins:
        msg_lines.append("✅ No high-impact overnight binary filings or negative news detected.")
        msg_lines.append("Market baseline remains clean for pre-open.")

    msg_lines.append("\n_Next Scan: 5:00 PM IST EOD Trade Recommendation Cycle_")
    bulletin_text = "\n".join(msg_lines)

    logger.info(f"\n{bulletin_text}")

    if telegram_bot.is_configured:
        logger.info("Dispatching Pre-Market News Bulletin to Telegram...")
        await telegram_bot.send_message(bulletin_text)

    logger.info(f"✅ 8:00 AM Pre-Market News Scan complete for {target_date}.")
    return 0


def main():
    args = parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    exit_code = asyncio.run(execute_premarket_news_scan(target, force=args.force))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
