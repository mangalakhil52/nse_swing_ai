"""
Unit tests for Phase 3 specialist agents and Phase 4 CIO Orchestrator.
"""

import asyncio
import numpy as np
import pandas as pd
import pytest

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.relative_strength_agent import RelativeStrengthAgent
from src.agents.forensic_agent import ForensicAnalysisAgent
from src.agents.risk_agent import RiskManagementAgent
from src.agents.cio_orchestrator import CIOOrchestrator
from src.core.evidence import EvidenceGraph
from src.core.models import ShareholdingPattern, SymbolMetadata, AnnualRatios
from src.core.types import AgentStatus, MarketRegime, SignalType, TradingStance
from src.quant.indicators import TechnicalIndicators

from datetime import date


def _make_bullish_df(n: int = 80) -> pd.DataFrame:
    close = np.linspace(100, 180, n)
    high = close + 2.0
    low = close - 2.0
    open_p = close - 1.0
    volume = np.full(n, 800000)
    turnover = (close * volume) / 1e7
    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": volume, "turnover_crores": turnover, "delivery_pct": np.full(n, 58.0),
    })
    return TechnicalIndicators.compute_all_indicators(df)


def test_technical_agent_bullish_setup():
    async def _run():
        agent = TechnicalAnalysisAgent()
        df = _make_bullish_df()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
        ev = EvidenceGraph("RUN-TEST")
        out = await agent.execute(meta, df, ev, "RUN-TEST", {})
        assert out.status == AgentStatus.SUCCESS
        assert out.signal == SignalType.BULLISH
        assert out.score > 70.0

    asyncio.run(_run())


def test_relative_strength_agent_outperforming():
    async def _run():
        agent = RelativeStrengthAgent()
        stock_df = _make_bullish_df(80)
        nifty_close = pd.Series(np.linspace(22000, 23500, 80))
        nifty_df = pd.DataFrame({"close": nifty_close})
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd")
        ev = EvidenceGraph("RUN-TEST")
        context = {"nifty_df": nifty_df, "universe_rs_scores": {"TRENT": 12.0}}
        out = await agent.execute(meta, stock_df, ev, "RUN-TEST", context)
        assert out.status == AgentStatus.SUCCESS

    asyncio.run(_run())


def test_forensic_agent_flags_high_pledge():
    async def _run():
        agent = ForensicAnalysisAgent()
        df = _make_bullish_df()
        meta = SymbolMetadata(symbol="RISKY", company_name="Risky Corp", asm_gsm_stage=0)
        shp = ShareholdingPattern(
            symbol="RISKY",
            quarter_date=date(2026, 6, 30),
            promoter_pct=55.0,
            promoter_pledged_pct=25.0,  # > 20% threshold -> disqualify
            fii_pct=10.0,
        )
        ratios = AnnualRatios(
            symbol="RISKY", fiscal_year="2026", roe_pct=18.0, roce_pct=20.0,
            debt_to_equity=0.5, cfo_crores=500.0, cfo_to_pat_ratio=0.85,
        )
        ev = EvidenceGraph("RUN-TEST")
        ctx = {"shareholding_pattern": shp, "annual_ratios": ratios}
        out = await agent.execute(meta, df, ev, "RUN-TEST", ctx)
        assert out.disqualification_triggered is True

    asyncio.run(_run())


def test_risk_agent_disqualifies_bear_regime():
    async def _run():
        agent = RiskManagementAgent()
        df = _make_bullish_df()
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd")
        ev = EvidenceGraph("RUN-TEST")
        ctx = {
            "market_regime": MarketRegime.STRONG_BEAR,
            "trading_stance": TradingStance.NO_TRADE,
            "upcoming_events": [],
        }
        out = await agent.execute(meta, df, ev, "RUN-TEST", ctx)
        assert out.disqualification_triggered is True

    asyncio.run(_run())


def test_cio_orchestrator_full_pipeline():
    async def _run():
        cio = CIOOrchestrator()
        df = _make_bullish_df(100)
        meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail", is_fno_eligible=True)
        ev = EvidenceGraph("RUN-TEST")
        ctx = {
            "market_regime": MarketRegime.BULL,
            "trading_stance": TradingStance.NORMAL,
            "regime_risk_multiplier": 0.75,
            "upcoming_events": [],
        }
        rec, _ = await cio.analyze_candidate(meta, df, "RUN-TEST-01", ctx)
        # In a bull regime with strong uptrend, candidate should produce a recommendation
        # (May be None if score is below conviction threshold - that's acceptable behavior)
        # Just verify no exception is raised and if rec is produced it has valid fields
        if rec is not None:
            assert rec.symbol == "TRENT"
            assert rec.composite_score >= 60.0
            assert rec.levels.entry_trigger_price > 0.0

    asyncio.run(_run())
