#!/usr/bin/env python3
"""
Pre-Market 8:00 AM News, Global Markets & Day Movement Scan — scripts/run_premarket_news_scan.py

Runs automatically every trading day at 08:00 AM IST (before NSE pre-open at 09:00 AM):
  1. Checks if today is an official NSE trading day.
  2. Fetches live pre-market GIFT Nifty, US Markets (S&P 500, Nasdaq, Dow), Asian Markets (Nikkei, Hang Seng),
     Brent Crude Oil, and USD/INR exchange rates.
  3. Ingests overnight corporate announcements, board meeting outcomes, order wins, and major stock news across Indian equities.
  4. Generates an AI Pre-Market Day Outlook & Market Movement Analysis for the day.
  5. Dispatches the 8:00 AM Pre-Market Bulletin to Telegram.

Usage:
  python scripts/run_premarket_news_scan.py [--run-now] [--force]
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.market_hours import MarketCalendar
from src.agents.catalyst_agent import CatalystAgent
from src.agents.news_agent import NewsIntelligenceAgent
from src.core.evidence import EvidenceGraph
from src.data.global_markets import GlobalMarketProvider
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
    parser = argparse.ArgumentParser(description="NSE Swing AI 8:00 AM Pre-Market News & Global Markets Scan")
    parser.add_argument("--run-now", action="store_true", help="Execute pre-market scan immediately")
    parser.add_argument("--force", action="store_true", help="Force scan execution even on weekends/holidays")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    return parser.parse_args()


async def execute_premarket_news_scan(target_date: date | None = None, force: bool = False) -> int:
    """Executes the 8:00 AM Pre-Market News, Global Markets & Day Movement Analysis cycle."""
    target_date = target_date or date.today()
    run_id = f"NEWS-8AM-{target_date.strftime('%Y%m%d')}-{int(datetime.now().timestamp())}"

    logger.info(f"{'='*60}")
    logger.info(f"NSE SWING AI — 8:00 AM PRE-MARKET GLOBAL MARKETS & NEWS BULLETIN")
    logger.info(f"Run ID: {run_id} | Date: {target_date} | Force: {force}")
    logger.info(f"{'='*60}")

    if not force and not MarketCalendar.is_trading_day(target_date):
        logger.info(f"{target_date} is a weekend or NSE market holiday. Skipping pre-market scan. Use --force to override.")
        return 0

    init_db()
    global_provider = GlobalMarketProvider()
    news_provider = FinancialNewsProvider()
    telegram_bot = TelegramBotNotifier()
    news_agent = NewsIntelligenceAgent()
    catalyst_agent = CatalystAgent()

    # 1. Fetch Global Markets & GIFT Nifty Cues
    logger.info("Fetching GIFT Nifty, US, Asian markets, Crude Oil & USD/INR pre-market metrics...")
    global_data = await global_provider.fetch_global_indices()
    gift = global_data["gift_nifty"]
    sp500 = global_data["sp500"]
    nasdaq = global_data["nasdaq"]
    nikkei = global_data["nikkei"]
    hang_seng = global_data["hang_seng"]
    crude = global_data["brent_crude"]
    usdinr = global_data["usdinr"]

    # 2. Ingest Major Indian Stock News & Announcements
    nse_provider = NseDataProvider()
    universe_engine = UniverseDiscoveryEngine(market_data_provider=nse_provider)
    universe_meta = await universe_engine.build_universe()
    await nse_provider.close()

    evidence_graph = EvidenceGraph(run_id=run_id)
    catalyst_alerts = []
    news_bulletins = []
    pos_news_count = 0
    neg_news_count = 0

    logger.info(f"Ingesting overnight corporate announcements and major stock news...")

    for meta in universe_meta[:80]:  # Focus on top 80 liquid Nifty market leaders
        sym = meta.symbol
        try:
            announcements = await news_provider.fetch_company_announcements(sym, lookback_days=2)
            articles = await news_provider.fetch_latest_news(sym, lookback_days=2)

            if not announcements and not articles:
                continue

            ctx = {"announcements": announcements, "news_articles": articles}
            import pandas as pd
            df_dummy = pd.DataFrame()

            news_out = await news_agent.execute(meta, df_dummy, evidence_graph, run_id, ctx)
            catalyst_out = await catalyst_agent.execute(meta, df_dummy, evidence_graph, run_id, ctx)

            if catalyst_out.score >= 70.0 or news_out.signal.value == "BULLISH":
                pos_news_count += 1
                desc = catalyst_out.metrics.get("description", "Positive Corporate Filing")
                catalyst_alerts.append(
                    f"⚡ *{sym}* ({meta.company_name}) | Score: *{catalyst_out.score:.0f}/100*\n"
                    f"   • {desc}"
                )

            elif news_out.signal.value == "BEARISH":
                neg_news_count += 1
                risk_msg = news_out.risks_identified[0] if news_out.risks_identified else "Negative press headline"
                news_bulletins.append(
                    f"⚠️ *{sym}* ({meta.company_name})\n"
                    f"   • {risk_msg}"
                )

        except Exception as e:
            logger.debug(f"Error evaluating news for {sym}: {e}")

    # 3. Generate Day Market Movement Analysis
    overall_news_sentiment = "BULLISH" if pos_news_count > neg_news_count else ("BEARISH" if neg_news_count > pos_news_count else "NEUTRAL")
    outlook = global_provider.generate_day_outlook(global_data, overall_news_sentiment)

    # 4. Format 8:00 AM Telegram Bulletin
    msg_lines = [
        f"🌅 *NSE SWING AI — 8:00 AM PRE-MARKET BULLETIN*",
        f"📅 Date: {target_date.strftime('%Y-%m-%d')} | Session: *Pre-Market*",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🌏 *GLOBAL MARKETS & GIFT NIFTY CUES*",
        f"  • *GIFT Nifty*: *{gift['price']:,.0f}* ({gift['change_pts']:+.1f} pts / *{gift['change_pct']:+.2f}%*)",
        f"  • *US Markets*: S&P 500 {sp500['change_pct']:+.2f}% | Nasdaq *{nasdaq['change_pct']:+.2f}%* | Dow {global_data['dow']['change_pct']:+.2f}%",
        f"  • *Asian Markets*: Nikkei {nikkei['change_pct']:+.2f}% | Hang Seng {hang_seng['change_pct']:+.2f}%",
        f"  • *Commodities & FX*: Brent Crude ${crude['price']:.1f} ({crude['change_pct']:+.2f}%) | USD/INR ₹{usdinr['price']:.2f}",
        f"",
        f"🎯 *EXPECTED OPENING & DAY MOVEMENT ANALYSIS*",
        f"  • *Opening Guidance*: *{outlook['expected_gap']}*",
        f"  • *Day Analysis*: {outlook['movement_analysis']}",
        f"  • *Key Strategy*: {outlook['key_strategy']}",
        f"",
    ]

    if catalyst_alerts:
        msg_lines.append(f"📰 *MAJOR STOCKS NEWS & OVERNIGHT FILINGS ({len(catalyst_alerts)}):*")
        for alert in catalyst_alerts[:6]:
            msg_lines.append(alert)
        msg_lines.append("")

    if news_bulletins:
        msg_lines.append(f"⚠️ *RISK WARNINGS & NEGATIVE NEWS ({len(news_bulletins)}):*")
        for warn in news_bulletins[:4]:
            msg_lines.append(warn)
        msg_lines.append("")

    if not catalyst_alerts and not news_bulletins:
        msg_lines.append("✅ No high-impact binary filings or negative news across major stocks.")
        msg_lines.append("Market baseline remains clean for pre-open.")

    msg_lines.extend([
        f"",
        f"_Next Scan: 5:00 PM IST EOD Trade Recommendation Cycle_",
    ])

    bulletin_text = "\n".join(msg_lines)
    logger.info(f"\n{bulletin_text}")

    # 5. Dispatch Telegram Bulletin
    if telegram_bot.is_configured:
        logger.info("Dispatching 8:00 AM Pre-Market News Bulletin to Telegram...")
        await telegram_bot.send_message(bulletin_text)

    logger.info(f"✅ 8:00 AM Pre-Market News & Global Scan complete for {target_date}.")
    return 0


def main():
    args = parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    exit_code = asyncio.run(execute_premarket_news_scan(target, force=args.force))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
