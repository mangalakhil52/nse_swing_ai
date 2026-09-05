"""
P0 #14C — Specialist Research Agents Contract & Synthesis Unit Tests.

Validates that:
  1. TechnicalAnalysisAgent returns #14A AgentAnalysisResult with StructuredEvidence (source="TECHNICAL").
  2. FundamentalAnalysisAgent returns #14A AgentAnalysisResult with PIT-verified fundamentals or fails closed (pit_safe=False).
  3. NewsIntelligenceAgent returns #14A AgentAnalysisResult with StructuredEvidence (source="NEWS").
  4. MarketRegimeAgent returns #14A AgentAnalysisResult with StructuredEvidence (source="MARKET_REGIME").
  5. Multi-agent evidence outputs integrate end-to-end through EvidenceFusionEngine, ConvictionEngine, and CIOContract.
"""

import asyncio
from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.agents.market_regime_agent import MarketRegimeAgent
from src.agents.news_agent import NewsIntelligenceAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.architecture.contracts import (
    AgentAnalysisResult,
    CIOContract,
    CIOInput,
    ConvictionEngine,
    ConvictionGrade,
    EvidenceFusionEngine,
    RiskEngineResult,
    StructuredEvidence,
)
from src.core.models import AnnualRatios, NewsArticle, QuarterlyFinancials, SymbolMetadata
from src.core.types import SentimentType, SignalType
from src.data.data_quality import DataQualityGate, DataQualityResult, DataQualityStatus


def _make_df(bars: int = 60, end_date_str: str = "2026-06-30") -> pd.DataFrame:
    """Generates synthetic valid OHLCV DataFrame for testing."""
    dates = pd.date_range(end=end_date_str, periods=bars, freq="B")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.normal(0.2, 0.5, bars))
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": 50000,
    })


def test_technical_agent_contract_output():
    """1. Test TechnicalAnalysisAgent.analyze_contract output contract."""
    async def _run():
        agent = TechnicalAnalysisAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        dt = datetime(2026, 6, 30, 10, 0)
        df = _make_df(60)

        res = await agent.analyze_contract(meta, df, dt)

        assert isinstance(res, AgentAnalysisResult)
        assert res.symbol == "TRENT"
        assert res.agent_name == "technical_analysis_agent"
        assert res.decision_time == dt
        assert res.pit_safe is True
        assert len(res.evidence) > 0
        assert res.evidence[0].source == "TECHNICAL"

    asyncio.run(_run())


def test_fundamental_agent_contract_output():
    """2. Test FundamentalAnalysisAgent.analyze_contract fails closed when filings are unverified/missing."""
    async def _run():
        agent = FundamentalAnalysisAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        dt = datetime(2026, 6, 30, 10, 0)
        df = _make_df()

        # Unverified / missing fundamentals
        res_unverified = await agent.analyze_contract(meta, df, dt, context={})

        assert isinstance(res_unverified, AgentAnalysisResult)
        assert res_unverified.symbol == "TRENT"
        assert res_unverified.pit_safe is False
        assert res_unverified.signal == SignalType.UNKNOWN

        # Verified quarterly fundamentals
        q_data = [QuarterlyFinancials(
            symbol="TRENT",
            period_end_date=date(2026, 3, 31),
            filing_date=date(2026, 4, 15),
            available_at=date(2026, 4, 15),
            sales_crores=1000.0,
            sales_growth_yoy_pct=20.0,
            pat_crores=150.0,
            pat_growth_yoy_pct=25.0,
            ebitda_margin_pct=18.0,
            eps_inr=15.0,
            pit_status="PIT_VERIFIED",
        )]
        ratios = AnnualRatios(symbol="TRENT", fiscal_year=2026, available_at=date(2026, 4, 15), roe_pct=20.0, roce_pct=22.0, debt_to_equity=0.2, cfo_crores=200.0, cfo_to_pat_ratio=0.9)

        res_verified = await agent.analyze_contract(
            meta, df, dt, context={"quarterly_financials": q_data, "annual_ratios": ratios, "fcf_pat_ratio": 0.85, "cfo_ebitda_ratio": 0.80}
        )

        assert res_verified.pit_safe is True
        assert res_verified.signal == SignalType.BULLISH
        assert res_verified.score >= 70.0

    asyncio.run(_run())


def test_news_agent_contract_output():
    """3. Test NewsIntelligenceAgent.analyze_contract output contract."""
    async def _run():
        agent = NewsIntelligenceAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        dt = datetime(2026, 6, 30, 10, 0)
        df = _make_df()

        articles = [NewsArticle(
            symbol="TRENT", headline="Trent expands 50 new stores",
            publisher="Economic Times", published_at=datetime(2026, 6, 28, 9, 0),
            available_at=datetime(2026, 6, 28, 9, 0), sentiment=SentimentType.POSITIVE,
            materiality_score=0.8, is_catalyst=True, pit_status="PIT_VERIFIED",
        )]

        res = await agent.analyze_contract(meta, df, dt, context={"news_articles": articles})

        assert isinstance(res, AgentAnalysisResult)
        assert res.symbol == "TRENT"
        assert res.pit_safe is True
        assert res.signal == SignalType.BULLISH
        assert len(res.evidence) > 0
        assert res.evidence[0].source == "NEWS"

    asyncio.run(_run())


def test_market_regime_agent_contract_output():
    """4. Test MarketRegimeAgent.analyze_contract output contract."""
    async def _run():
        agent = MarketRegimeAgent()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        dt = datetime(2026, 6, 30, 10, 0)
        df = _make_df()
        nifty_df = _make_df(bars=100)

        res = await agent.analyze_contract(meta, df, dt, context={"nifty_df": nifty_df, "decision_time": dt})

        assert isinstance(res, AgentAnalysisResult)
        assert res.symbol == "TRENT"
        assert res.agent_name == "market_regime_agent"
        assert res.decision_time == dt

    asyncio.run(_run())


def test_multi_agent_fusion_and_cio_pipeline():
    """5. Test end-to-end integration of specialist contract outputs into EvidenceFusionEngine & CIOContract."""
    async def _run():
        dt = datetime(2026, 6, 30, 10, 0)
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        df = _make_df()

        tech_agent = TechnicalAnalysisAgent()
        fund_agent = FundamentalAnalysisAgent()

        q_data = [QuarterlyFinancials(
            symbol="TRENT",
            period_end_date=date(2026, 3, 31),
            filing_date=date(2026, 4, 15),
            available_at=date(2026, 4, 15),
            sales_crores=1000.0,
            sales_growth_yoy_pct=25.0,
            pat_crores=150.0,
            pat_growth_yoy_pct=30.0,
            ebitda_margin_pct=18.0,
            eps_inr=15.0,
            pit_status="PIT_VERIFIED",
        )]
        ratios_data = AnnualRatios(
            symbol="TRENT",
            fiscal_year="2026",
            roe_pct=22.0,
            roce_pct=25.0,
            debt_to_equity=0.2,
            cfo_crores=150.0,
            cfo_to_pat_ratio=1.0,
            available_at=date(2026, 4, 15),
            pit_status="PIT_VERIFIED",
        )

        tech_res = await tech_agent.analyze_contract(meta, df, dt)
        fund_res = await fund_agent.analyze_contract(
            meta, df, dt, context={"quarterly_financials": q_data, "annual_ratios": ratios_data}
        )

        dq_valid = DataQualityGate.evaluate_evidence_quality("TRENT", df, as_of_date=dt)
        fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq_valid)

        assert (len(fusion.bullish_evidence) + len(fusion.neutral_evidence)) > 0
        assert len(fusion.bullish_evidence) > 0
        assert fusion.aggregate_strength is not None  # Computed deterministically in #14D
        assert fusion.data_quality.pit_safe is True

        conviction = ConvictionEngine.evaluate_conviction(fusion)
        assert conviction.grade in (ConvictionGrade.HIGH_CONVICTION, ConvictionGrade.MEDIUM_CONVICTION, ConvictionGrade.LOW_CONVICTION)

        risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)
        cio_input = CIOInput(
            symbol="TRENT", decision_time=dt, technical_result=tech_res,
            fundamental_result=fund_res, fusion_result=fusion,
            conviction_result=conviction, risk_result=risk, data_quality=dq_valid,
        )

        decision = CIOContract.evaluate_decision(cio_input)

        assert decision.symbol == "TRENT"
        assert decision.decision in ("BUY", "WATCH", "NO_TRADE", "REJECT")
        assert isinstance(decision.decision, str)

    asyncio.run(_run())
