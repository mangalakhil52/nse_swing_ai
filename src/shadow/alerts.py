"""
Alert Dispatcher Module.
Formats and dispatches trade recommendations and shadow P&L updates via Telegram and Markdown reports.
"""

import logging
from datetime import date, datetime

from src.core.models import TradeRecommendation
from src.core.types import ConvictionGrade

logger = logging.getLogger(__name__)


class TelegramFormatter:
    """Formats production-grade Telegram messages for the paper/shadow scanner."""

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
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"*{rec.symbol}* | {rec.company_name}",
            f"Score: *{rec.composite_score:.1f} / 100* | Setup: *{rec.technical_setup_description}*",
            "💰 *TRADE LEVELS:*",
            f"  • Current Market Price: ₹{rec.levels.current_market_price:.2f}",
            f"  • Entry Trigger Price : ₹{rec.levels.entry_trigger_price:.2f}",
            f"  • Stop Loss Price     : ₹{rec.levels.stop_loss_price:.2f}",
            f"  • Target 1            : ₹{rec.levels.target_1:.2f}",
            f"  • Target 2            : ₹{rec.levels.target_2:.2f}",
            f"  • Target 3            : ₹{rec.levels.target_3:.2f}",
            "📦 *POSITION SIZING:*",
            f"  • Allocated Shares    : {rec.levels.position_size_shares} shares",
            f"  • Capital Allocated   : ₹{rec.levels.allocated_capital_rupees:,.0f}",
            f"🆔 Run ID: `{rec.run_id}`",
            "⚠️ *PAPER / SHADOW TRADE — NOT AN EXECUTION ORDER*",
        ]
        return "\n".join(lines)

    @classmethod
    def format_scan_summary(cls, recs: list[TradeRecommendation], regime: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        if not recs:
            return cls.format_no_trade(date.today(), "No qualifying swing trade setups detected", regime=regime)

        lines = [
            "🎯 *NSE SWING AI — DAILY RECOMMENDATIONS*",
            f"📅 Date: {today} | Regime: *{regime}*",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"*{len(recs)} Top Trade Setup(s) Found:*",
            "",
        ]
        for i, rec in enumerate(recs, 1):
            emoji = cls.CONVICTION_EMOJI.get(rec.conviction, "📋")
            lines.append(
                f"{i}. {emoji} *{rec.symbol}* ({rec.conviction.value}) | Score: {rec.composite_score:.1f} | "
                f"Entry: ₹{rec.levels.entry_trigger_price:.2f} | SL: ₹{rec.levels.stop_loss_price:.2f} | "
                f"T1: ₹{rec.levels.target_1:.2f}"
            )
        lines.append("\n_See individual alerts for full trade levels._")
        lines.append("⚠️ *PAPER / SHADOW TRADE — NOT AN EXECUTION ORDER*")
        return "\n".join(lines)

    @classmethod
    def format_no_trade(cls, scan_date: date, reason: str, *, regime: str | None = None, run_id: str | None = None) -> str:
        """Explicitly report a completed scan that intentionally produced no trade."""
        lines = [
            "🟦 *NSE SWING AI — NO TRADE TODAY*",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 Date: {scan_date.isoformat()}",
        ]
        if regime:
            lines.append(f"📊 Market Regime: *{regime}*")
        lines.extend([
            f"Reason: *{reason}*",
            "",
            "No position should be opened from today's scan.",
        ])
        if run_id:
            lines.append(f"🆔 Run ID: `{run_id}`")
        lines.append("⚠️ *PAPER / SHADOW MODE*")
        return "\n".join(lines)

    @classmethod
    def format_system_status(cls, scan_date: date, reason: str, *, run_id: str | None = None) -> str:
        """Report a failed/blocked scan separately from a valid no-trade result."""
        lines = [
            "🚨 *NSE SWING AI — SCAN STATUS*",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 Date: {scan_date.isoformat()}",
            f"Status: *SYSTEM ERROR / DATA UNAVAILABLE*",
            f"Reason: `{reason}`",
            "",
            "⚠️ *NO TRADE — DO NOT ACT ON THIS ALERT*",
        ]
        if run_id:
            lines.append(f"🆔 Run ID: `{run_id}`")
        return "\n".join(lines)


class MarkdownReportWriter:
    """Writes structured Markdown research dossiers for each trade recommendation."""

    @classmethod
    def write_recommendation_dossier(cls, rec: TradeRecommendation, output_path: str) -> str:
        lines = [
            f"# {rec.symbol} — Swing Trade Dossier",
            f"**Date**: {rec.recommendation_date} | **Run ID**: {rec.run_id}",
            f"**Conviction**: {rec.conviction.value} | **Score**: {rec.composite_score:.1f}/100",
            "",
            "## Trade Levels",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| CMP | ₹{rec.levels.current_market_price:.2f} |",
            f"| Entry Trigger | ₹{rec.levels.entry_trigger_price:.2f} |",
            f"| Stop Loss | ₹{rec.levels.stop_loss_price:.2f} ({rec.levels.risk_percentage:.1f}%) |",
            f"| Target 1 | ₹{rec.levels.target_1:.2f} (R:R {rec.levels.risk_reward_t1:.1f}) |",
            f"| Target 2 | ₹{rec.levels.target_2:.2f} (R:R {rec.levels.risk_reward_t2:.1f}) |",
            f"| Target 3 | ₹{rec.levels.target_3:.2f} (R:R {rec.levels.risk_reward_t3:.1f}) |",
            f"| Shares | {rec.levels.position_size_shares} |",
            f"| Capital Allocated | ₹{rec.levels.allocated_capital_rupees:,.0f} |",
            "",
            "## Technical Setup",
            rec.technical_setup_description,
            "",
            "## Why This Trade",
        ]
        for why in rec.why_this_trade:
            lines.append(f"- {why}")

        lines.extend([
            "",
            "## Fundamental Summary",
            rec.fundamental_summary,
            "",
            "## Catalyst",
            rec.catalyst_summary,
            "",
            "## Sector Context",
            rec.sector_context,
            "",
            "## Market Regime",
            rec.market_regime,
            "",
            "## Major Risks",
        ])
        for risk in rec.major_risks:
            lines.append(f"- ⚠️ {risk}")

        lines.extend(["", "## Invalidation", rec.invalidation_rules, "", "## Evidence Dossier"])
        for ev in rec.evidence_dossier[:10]:
            lines.append(f"- **{ev.metric_name}**: {ev.observed_value} *(Source: {ev.source})*")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content
