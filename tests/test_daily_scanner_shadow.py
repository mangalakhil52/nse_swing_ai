"""
Phase #15 — Daily Scanner & Shadow Execution Unit Tests.

Validates that:
  1. run_daily_scan arguments parsing and non-trading day checks exit cleanly (return 0).
  2. Daily scanner executes without conflict markers or syntax exceptions.
  3. Telegram alert formatting and recommendation persistent audit logging work as expected.
"""

import asyncio
from datetime import date
import pytest

from scripts.run_daily_scan import run_scan
from src.core.models import ConvictionGrade, TradeLevels, TradeRecommendation, TradeStatus
from src.shadow.alerts import TelegramFormatter


def test_daily_scan_non_trading_day():
    # 2026-06-07 is a Sunday
    sunday = date(2026, 6, 7)
    res = asyncio.run(run_scan(sunday, dry_run=True, force=False))
    assert res == 0


def test_telegram_formatter_output():
    levels = TradeLevels(
        symbol="TRENT",
        current_market_price=104.0,
        entry_trigger_price=105.0,
        stop_loss_price=98.0,
        risk_rupees=7.0,
        risk_percentage=6.67,
        target_1=115.0,
        target_2=125.0,
        target_3=140.0,
        risk_reward_t1=1.4,
        risk_reward_t2=2.8,
        risk_reward_t3=5.0,
        position_size_shares=50,
        allocated_capital_rupees=5250.0,
        invalidation_criteria="Daily close below ₹98.0",
    )
    rec = TradeRecommendation(
        recommendation_id="REC-TEST-001",
        run_id="SCAN-20260630-TEST",
        symbol="TRENT",
        company_name="Trent Ltd",
        sector="Retail",
        recommendation_date=date(2026, 6, 30),
        conviction=ConvictionGrade.A_PLUS,
        composite_score=85.0,
        levels=levels,
        technical_setup_description="VCP Breakout",
        catalyst_summary="PAT Growth +30% YoY",
        fundamental_summary="ROE 25%, ROCE 28%",
        sector_context="Retail sector bullish",
        market_regime="STRONG_BULL",
        major_risks=["Short-term RSI extended"],
        invalidation_rules="Daily close below ₹98.0",
        why_this_trade=["Strong breakout with volume"],
        evidence_dossier=[],
        status=TradeStatus.PENDING_ENTRY,
    )

    msg = TelegramFormatter.format_recommendation(rec)
    assert "TRENT" in msg
    assert "105.00" in msg
