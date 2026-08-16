"""
Alert Dispatcher Module.
Formats and dispatches trade recommendations and shadow P&L updates via Telegram and Markdown reports.
"""

import logging
from datetime import datetime

from src.core.models import TradeRecommendation
from src.core.types import ConvictionGrade

logger = logging.getLogger(__name__)


class TelegramFormatter:
    """Formats TradeRecommendation into production-grade Telegram Markdown messages."""

    CONVICTION_EMOJI = {
        ConvictionGrade.A_PLUS: "🔥",
        ConvictionGrade.A: "✅",
        ConvictionGrade.B_PLUS: "🟡",
        ConvictionGrade.B: "🟠",
        ConvictionGrade.C: "⚪",
        ConvictionGrade.REJECT: "❌",
    }

    @classmethod
    def format_recommendation(cls, rec: TradeRecommendation) -> str:
        emoji = cls.CONVICTION_EMOJI.get(rec.conviction, "📋")

        lines = [
            f"{emoji} *NSE SWING TRADE — {rec.conviction.value} CONVICTION*",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"*{rec.symbol}* | {rec.company_name}",
            f"Score: *{rec.composite_score:.1f} / 100* | Setup: *{rec.technical_setup_description}*",
            f"💰 *TRADE LEVELS:*",
            f"  • Current Market Price: ₹{rec.levels.current_market_price:.2f}",
            f"  • Entry Trigger Price : ₹{rec.levels.entry_trigger_price:.2f} (Place Buy Stop Order / Trigger above this price)",
            f"  • Stop Loss Price     : ₹{rec.levels.stop_loss_price:.2f} (Exit immediately if daily close < ₹{rec.levels.stop_loss_price:.2f})",
            f"  • Target 1            : ₹{rec.levels.target_1:.2f} (Exit 50% shares here & move stop to cost)",
            f"  • Target 2            : ₹{rec.levels.target_2:.2f} (Exit 30% shares here)",
            f"  • Target 3            : ₹{rec.levels.target_3:.2f} (Trail remaining 20% to Target 3)",
            f"📦 *POSITION SIZING:*",
            f"  • Allocated Shares    : {rec.levels.position_size_shares} shares",
            f"  • Capital Allocated   : ₹{rec.levels.allocated_capital_rupees:,.0f}",
        ]

        return "\n".join(lines)

    @classmethod
    def format_scan_summary(cls, recs: list[TradeRecommendation], regime: str) -> str:
        if not recs:
            return (
                f"📋 *DAILY SCAN COMPLETE — {datetime.utcnow().strftime('%Y-%m-%d')}*\n"
                f"Market Regime: *{regime}*\n\n"
                f"⚠️ No qualifying swing trade setups detected today.\n"
                f"Maintain existing positions. Watch for better entries."
            )

        lines = [
            f"🎯 *NSE SWING AI — DAILY RECOMMENDATIONS*",
            f"📅 Date: {datetime.utcnow().strftime('%Y-%m-%d')} | Regime: *{regime}*",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"*{len(recs)} Top Trade Setup(s) Found:*",
            f"",
        ]
        for i, rec in enumerate(recs, 1):
            emoji = cls.CONVICTION_EMOJI.get(rec.conviction, "📋")
            lines.append(
                f"{i}. {emoji} *{rec.symbol}* ({rec.conviction.value}) | Score: {rec.composite_score:.1f} | "
                f"Entry: ₹{rec.levels.entry_trigger_price:.2f} | SL: ₹{rec.levels.stop_loss_price:.2f} | "
                f"T1: ₹{rec.levels.target_1:.2f}"
            )

        lines.append(f"\n_See individual alerts for full trade levels._")
        return "\n".join(lines)


class MarkdownReportWriter:
    """Writes structured Markdown research dossiers for each trade recommendation."""

    @classmethod
    def write_recommendation_dossier(cls, rec: TradeRecommendation, output_path: str) -> str:
        """Generates a detailed Markdown research dossier."""
        lines = [
            f"# {rec.symbol} — Swing Trade Dossier",
            f"**Date**: {rec.recommendation_date} | **Run ID**: {rec.run_id}",
            f"**Conviction**: {rec.conviction.value} | **Score**: {rec.composite_score:.1f}/100",
            f"",
            f"## Trade Levels",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| CMP | ₹{rec.levels.current_market_price:.2f} |",
            f"| Entry Trigger | ₹{rec.levels.entry_trigger_price:.2f} |",
            f"| Stop Loss | ₹{rec.levels.stop_loss_price:.2f} ({rec.levels.risk_percentage:.1f}%) |",
            f"| Target 1 | ₹{rec.levels.target_1:.2f} (R:R {rec.levels.risk_reward_t1:.1f}) |",
            f"| Target 2 | ₹{rec.levels.target_2:.2f} (R:R {rec.levels.risk_reward_t2:.1f}) |",
            f"| Target 3 | ₹{rec.levels.target_3:.2f} (R:R {rec.levels.risk_reward_t3:.1f}) |",
            f"| Shares | {rec.levels.position_size_shares} |",
            f"| Capital Allocated | ₹{rec.levels.allocated_capital_rupees:,.0f} |",
            f"",
            f"## Technical Setup",
            f"{rec.technical_setup_description}",
            f"",
            f"## Why This Trade",
        ]
        for why in rec.why_this_trade:
            lines.append(f"- {why}")

        lines.extend([
            f"",
            f"## Fundamental Summary",
            f"{rec.fundamental_summary}",
            f"",
            f"## Catalyst",
            f"{rec.catalyst_summary}",
            f"",
            f"## Sector Context",
            f"{rec.sector_context}",
            f"",
            f"## Market Regime",
            f"{rec.market_regime}",
            f"",
            f"## Major Risks",
        ])
        for risk in rec.major_risks:
            lines.append(f"- ⚠️ {risk}")

        lines.extend([
            f"",
            f"## Invalidation",
            f"{rec.invalidation_rules}",
            f"",
            f"## Evidence Dossier",
        ])
        for ev in rec.evidence_dossier[:10]:
            lines.append(f"- **{ev.metric_name}**: {ev.observed_value} *(Source: {ev.source})*")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content
