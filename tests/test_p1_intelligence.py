"""
Unit tests for P1 Intelligence Engine Upgrades:
  1. Upgraded News Intelligence Agent (Materiality, Surprise, Unpriced Catalysts)
  2. Upgraded Fundamental Analysis Agent (Earnings Acceleration, Cash Quality FCF/PAT)
  3. Thesis Killer Agent (Devil's Advocate Fragility Score & Veto)
  4. Probability-of-Path & Expected Value Engine (P(Win) & EV)
  5. Execution-Quality & Market Impact Slippage Model
"""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import pytest

from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.agents.news_agent import NewsIntelligenceAgent
from src.agents.thesis_killer_agent import ThesisKillerAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    AnnualRatios,
    NewsArticle,
    QuarterlyFinancials,
    SymbolMetadata,
)
from src.core.types import (
    CatalystType,
    MarketRegime,
    PatternType,
    SentimentType,
    SignalType,
    SourceTier,
)
from src.quant.probability_engine import ProbabilityPathEngine
from src.risk.execution_quality import ExecutionQualityModel


def test_upgraded_news_agent():
    async def _run():
        agent = NewsIntelligenceAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        ev = EvidenceGraph("TEST-RUN")
        articles = [
            NewsArticle(
                symbol="TRENT",
                headline="Trent Q1 PAT beats consensus by +18%",
                summary="Strong sales momentum",
                publisher="Economic Times",
                source_tier=SourceTier.TIER_2,
                source_url="https://et.com",
                published_at=datetime.utcnow() - timedelta(days=1),
                sentiment=SentimentType.POSITIVE,
                materiality_score=0.88,
                is_catalyst=True,
                catalyst_type=CatalystType.EARNINGS_ANNOUNCEMENT,
            )
        ]
        ctx = {"news_articles": articles, "announcements": [], "earnings_surprise_pct": 18.0}
        out = await agent.execute(meta, pd.DataFrame(), ev, "TEST-RUN", ctx)

        assert out.status.value == "SUCCESS"
        assert out.signal == SignalType.BULLISH
        assert out.score >= 80.0
        assert out.metrics["earnings_surprise_pct"] == 18.0
        assert out.metrics["unpriced_catalyst_count"] == 1

    asyncio.run(_run())


def test_upgraded_fundamental_agent():
    async def _run():
        agent = FundamentalAnalysisAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        ev = EvidenceGraph("TEST-RUN")
        q1 = QuarterlyFinancials(
            symbol="TRENT",
            period_end_date=date.today(),
            sales_crores=1200.0,
            pat_crores=180.0,
            ebitda_margin_pct=18.5,
            eps_inr=12.5,
            sales_growth_yoy_pct=22.0,
            pat_growth_yoy_pct=35.0,
        )
        q2 = QuarterlyFinancials(
            symbol="TRENT",
            period_end_date=date.today() - timedelta(days=90),
            sales_crores=1000.0,
            pat_crores=140.0,
            ebitda_margin_pct=17.0,
            eps_inr=10.0,
            sales_growth_yoy_pct=15.0,
            pat_growth_yoy_pct=20.0,
        )
        ratios = AnnualRatios(
            symbol="TRENT",
            fiscal_year=2026,
            roe_pct=22.0,
            roce_pct=25.0,
            debt_to_equity=0.25,
            cfo_crores=165.0,
            cfo_to_pat_ratio=0.92,
        )
        ctx = {
            "quarterly_financials": [q1, q2],
            "annual_ratios": ratios,
            "fcf_pat_ratio": 0.88,
            "cfo_ebitda_ratio": 0.82,
        }
        out = await agent.execute(meta, pd.DataFrame(), ev, "TEST-RUN", ctx)

        assert out.status.value == "SUCCESS"
        assert out.signal == SignalType.BULLISH
        assert out.metrics["earnings_acceleration"] == 15.0  # 35.0 - 20.0
        assert out.metrics["fcf_to_pat"] == 0.88

    asyncio.run(_run())


def test_thesis_killer_agent_veto():
    async def _run():
        agent = ThesisKillerAgent()
        meta = SymbolMetadata(symbol="WEAKCO", company_name="Weak Corp", sector="General")
        ev = EvidenceGraph("TEST-RUN")

        # Fake sub-agent output with low FCF/PAT and negative news
        fund_out = AgentOutput(
            agent_name="fundamental_analysis_agent",
            symbol="WEAKCO",
            run_id="TEST-RUN",
            metrics={"fcf_to_pat": 0.35, "earnings_acceleration": -25.0},
        )
        news_out = AgentOutput(
            agent_name="news_intelligence_agent",
            symbol="WEAKCO",
            run_id="TEST-RUN",
            metrics={"negative_articles": 2},
        )
        ctx = {
            "agent_outputs": {
                "fundamental_analysis_agent": fund_out,
                "news_intelligence_agent": news_out,
            }
        }
        out = await agent.execute(meta, pd.DataFrame(), ev, "TEST-RUN", ctx)

        assert out.disqualification_triggered is True
        assert "THESIS KILLED" in out.disqualification_reason

    asyncio.run(_run())


def test_probability_path_engine():
    from src.quant.probability_engine import HistoricalSetupOutcome, HistoricalSetupOutcomeStore
    # Register 40 empirical outcomes for test
    outcomes = [
        HistoricalSetupOutcome(
            symbol=f"STOCK_{i}",
            pattern_type=PatternType.VOLATILITY_CONTRACTION_PATTERN,
            market_regime=MarketRegime.STRONG_BULL,
            setup_date="2026-01-01",
            entry_price=100.0,
            stop_loss=95.0,
            target_1=110.0,
            t1_hit_before_sl=(i < 28),  # 28 wins out of 40 = 70% win rate
            holding_sessions=5,
            exit_date="2026-01-08",
            source="NSE_BHAVCOPY_DAILY",
        )
        for i in range(40)
    ]
    HistoricalSetupOutcomeStore.register_outcomes(outcomes, persist=False)

    res = ProbabilityPathEngine.evaluate_expectancy(
        pattern_type=PatternType.VOLATILITY_CONTRACTION_PATTERN,
        market_regime=MarketRegime.STRONG_BULL,
        mansfield_rs=12.5,
        target1_pct=14.0,
        stop_loss_pct=6.5,
        fcf_pat_ratio=0.88,
    )
    assert res.is_ev_positive is True
    assert res.win_probability == 0.70
    assert res.confidence_type == "EMPIRICAL"
    assert res.expected_value > 0.0


def test_execution_quality_model():
    res = ExecutionQualityModel.evaluate_execution_quality(
        current_price=500.0,
        entry_trigger_price=506.0,
        adtv_crores=45.0,
        allocated_capital_rupees=150000.0,
        atr_14=12.0,
    )
    assert res.is_executable is True
    assert res.expected_slippage_pct <= 0.30
    assert res.adjusted_entry_trigger >= 506.0
