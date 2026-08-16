"""
Unit tests for Phase 6: Shadow Monitor, Alert Formatter, and Markdown Report Writer.
"""

from datetime import date
import tempfile
import os

import pytest

from src.core.models import TradeLevels, TradeRecommendation
from src.core.types import ConvictionGrade, TradeStatus
from src.shadow.alerts import MarkdownReportWriter, TelegramFormatter
from src.shadow.monitor import ShadowPerformanceReport, ShadowTradeUpdate


def _make_sample_rec() -> TradeRecommendation:
    levels = TradeLevels(
        symbol="TRENT",
        current_market_price=7200.0,
        entry_trigger_price=7250.0,
        stop_loss_price=6950.0,
        risk_rupees=300.0,
        risk_percentage=4.14,
        target_1=7790.0,
        target_2=8090.0,
        target_3=8600.0,
        risk_reward_t1=1.8,
        risk_reward_t2=2.8,
        risk_reward_t3=4.5,
        position_size_shares=33,
        allocated_capital_rupees=239250.0,
        invalidation_criteria="Daily close below ₹6950",
    )
    return TradeRecommendation(
        recommendation_id="REC-TEST-001",
        run_id="SCAN-20260816-ABCD1234",
        symbol="TRENT",
        company_name="Trent Ltd",
        sector="Retail",
        recommendation_date=date(2026, 8, 16),
        conviction=ConvictionGrade.A_PLUS,
        composite_score=91.5,
        levels=levels,
        technical_setup_description="VCP Breakout with 3 contracting waves",
        catalyst_summary="Strong Q1 FY27 results expected",
        fundamental_summary="ROE 28%, ROCE 32%, D/E 0.12",
        sector_context="Retail sector ranked #2 of 14",
        market_regime="STRONG_BULL",
        major_risks=["Broad market correction risk"],
        invalidation_rules="Daily close below ₹6950",
        why_this_trade=["VCP pattern detected", "Outperforming NIFTY by 15%", "PAT growth +35% YoY"],
        evidence_dossier=[],
        status=TradeStatus.PENDING_ENTRY,
    )


def test_telegram_formatter():
    rec = _make_sample_rec()
    msg = TelegramFormatter.format_recommendation(rec)
    assert "TRENT" in msg
    assert "A+" in msg
    assert "₹7,250.00" in msg or "7250" in msg
    assert "₹6,950.00" in msg or "6950" in msg


def test_telegram_scan_summary():
    rec = _make_sample_rec()
    summary = TelegramFormatter.format_scan_summary([rec], "STRONG_BULL")
    assert "TRENT" in summary
    assert "STRONG_BULL" in summary

    empty_summary = TelegramFormatter.format_scan_summary([], "NEUTRAL")
    assert "No qualifying" in empty_summary


def test_shadow_trade_stop_loss():
    trade = {
        "symbol": "TRENT", "status": "ACTIVE",
        "entry_price": 7200.0, "stop_loss": 6950.0,
        "target_1": 7740.0, "target_2": 8040.0, "target_3": 8550.0,
    }
    # Bar where open gaps below stop
    bar = {"open": 6900.0, "high": 7050.0, "low": 6880.0, "close": 7000.0}
    updated = ShadowTradeUpdate.check_and_update(trade, bar, session_count=3)
    assert updated["status"] == "STOPPED_OUT"
    assert updated["exit_price"] == 6900.0
    assert (updated["pnl_pct"] or 0.0) < 0.0


def test_shadow_trade_target1_hit():
    trade = {
        "symbol": "TRENT", "status": "ACTIVE",
        "entry_price": 7200.0, "stop_loss": 6950.0,
        "target_1": 7740.0, "target_2": 8040.0, "target_3": 8550.0,
    }
    bar = {"open": 7700.0, "high": 7800.0, "low": 7690.0, "close": 7780.0}
    updated = ShadowTradeUpdate.check_and_update(trade, bar, session_count=5)
    assert updated.get("t1_hit") is True
    assert updated["status"] == "TARGET_1_HIT"


def test_shadow_performance_report():
    closed_trades = [
        {"pnl_pct": 12.5, "status": "TARGET_2_HIT"},
        {"pnl_pct": -4.1, "status": "STOPPED_OUT"},
        {"pnl_pct": 8.3, "status": "TARGET_1_HIT"},
    ]
    report = ShadowPerformanceReport.generate_report(closed_trades)
    assert report["total"] == 3
    assert report["winners"] == 2
    assert report["win_rate"] == pytest.approx(66.7, abs=0.2)
    assert report["avg_pnl_pct"] > 0.0


def test_markdown_report_writer():
    rec = _make_sample_rec()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        output_path = f.name

    try:
        content = MarkdownReportWriter.write_recommendation_dossier(rec, output_path)
        assert "TRENT" in content
        assert "₹7,250.00" in content or "7250" in content
        assert os.path.exists(output_path)
    finally:
        os.unlink(output_path)
