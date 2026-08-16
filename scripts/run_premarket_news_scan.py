#!/usr/bin/env python3
"""
Pre-Market 8:00 AM News, Global Markets & Day Movement Scan — scripts/run_premarket_news_scan.py

Runs automatically every trading day at 08:00 AM IST (before NSE pre-open at 09:00 AM):
  1. Checks if today is an official NSE trading day.
  2. Ingests GIFT Nifty, US Markets (S&P 500, Nasdaq), Nikkei, Crude Oil, and USD/INR.
  3. Ingests FII/DII net institutional flows in Cash segment.
  4. Categorizes overnight stock news: Deals/Orders, Earnings/Results, Insider Trades, and Risks.
  5. Formats a clean, mobile-optimized 8:00 AM Telegram Pre-Market Bulletin.

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
from src.data.global_markets import GlobalMarketProvider
from src.data.news_provider import FinancialNewsProvider
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
    logger.info(f"NSE SWING AI — 8:00 AM PRE-MARKET BULLETIN (MOBILE-OPTIMIZED)")
    logger.info(f"Run ID: {run_id} | Date: {target_date} | Force: {force}")
    logger.info(f"{'='*60}")

    if not force and not MarketCalendar.is_trading_day(target_date):
        logger.info(f"{target_date} is a weekend or NSE market holiday. Skipping pre-market scan. Use --force to override.")
        return 0

    init_db()
    global_provider = GlobalMarketProvider()
    news_provider = FinancialNewsProvider()
    telegram_bot = TelegramBotNotifier()

    # 1. Fetch Global Markets & GIFT Nifty Cues
    logger.info("Fetching GIFT Nifty, US/Asian indices, Crude Oil & USD/INR...")
    global_data = await global_provider.fetch_global_indices()
    gift = global_data["gift_nifty"]
    sp500 = global_data["sp500"]
    nasdaq = global_data["nasdaq"]
    nikkei = global_data["nikkei"]
    crude = global_data["brent_crude"]
    usdinr = global_data["usdinr"]

    # 2. Fetch FII / DII Institutional Flows
    fii_dii = await FinancialNewsProvider.fetch_fii_dii_flows()

    # 3. Categorize Major Stock News & Overnight Filings
    categorized_news = await news_provider.fetch_categorized_premarket_news()
    deals = categorized_news.get("deals", [])
    earnings = categorized_news.get("earnings", [])
    insider = categorized_news.get("insider_promoter", [])
    risks = categorized_news.get("risks", [])

    # 4. Generate Day Market Movement Analysis
    overall_sentiment = "BULLISH" if gift["change_pts"] >= 0 else "BEARISH"
    outlook = global_provider.generate_day_outlook(global_data, overall_sentiment)

    # 5. Format Clean Mobile-Optimized Telegram Message
    lines = [
        f"🌅 *8:00 AM PRE-MARKET BULLETIN*",
        f"📅 {target_date.strftime('%Y-%m-%d')} | NSE Pre-Market",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🌐 *GLOBAL CUES & GIFT NIFTY*",
        f"• *GIFT Nifty*: *{gift['price']:,.0f}* ({gift['change_pts']:+.1f} pts | *{gift['change_pct']:+.2f}%*)",
        f"• *US Markets*: S&P {sp500['change_pct']:+.2f}% | Nasdaq *{nasdaq['change_pct']:+.2f}%*",
        f"• *Asia & FX*: Nikkei {nikkei['change_pct']:+.2f}% | Crude ${crude['price']:.1f} ({crude['change_pct']:+.2f}%) | USD/INR ₹{usdinr['price']:.2f}",
        f"",
        f"🎯 *EXPECTED DAY MOVEMENT*",
        f"• *Open*: *{outlook['expected_gap']}*",
        f"• *Trend*: {outlook['gap_type'].replace('_', ' ')} Guidance",
        f"• *Strategy*: {outlook['key_strategy']}",
        f"",
        f"🏛️ *FII / DII CASH FLOWS*",
        f"• *FII Net*: +₹{fii_dii['fii_net_crores']:,.0f} Cr ({fii_dii['fii_action']})",
        f"• *DII Net*: +₹{fii_dii['dii_net_crores']:,.0f} Cr ({fii_dii['dii_action']})",
        f"",
        f"📰 *TOP STOCK NEWS & FILINGS*",
    ]

    if deals:
        lines.append(f"\n📁 *Deals & Order Wins:*")
        for item in deals[:3]:
            lines.append(f"• *{item['symbol']}*: {item['headline']}")

    if earnings:
        lines.append(f"\n📊 *Earnings & Results:*")
        for item in earnings[:3]:
            lines.append(f"• *{item['symbol']}*: {item['headline']}")

    if insider:
        lines.append(f"\n💼 *Insider & Promoter Trades:*")
        for item in insider[:2]:
            lines.append(f"• *{item['symbol']}*: {item['headline']}")

    if risks:
        lines.append(f"\n⚠️ *Risk Warnings:*")
        for item in risks[:2]:
            lines.append(f"• *{item['symbol']}*: {item['headline']}")
    else:
        lines.append(f"\n⚠️ *Risk Warnings:*\n• None detected for pre-open.")

    lines.extend([
        f"",
        f"_Next Scan: 5:00 PM IST EOD Cycle_",
    ])

    bulletin_text = "\n".join(lines)
    logger.info(f"\n{bulletin_text}")

    # 6. Dispatch to Telegram
    if telegram_bot.is_configured:
        logger.info("Dispatching mobile-optimized bulletin to Telegram...")
        await telegram_bot.send_message(bulletin_text)

    logger.info(f"✅ Mobile 8:00 AM Pre-Market News Bulletin complete for {target_date}.")
    return 0


def main():
    args = parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    exit_code = asyncio.run(execute_premarket_news_scan(target, force=args.force))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
