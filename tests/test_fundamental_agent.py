"""P0 #14D — Fundamental Agent contract, PIT, and evidence tests."""

import asyncio
from datetime import date
import pandas as pd

from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.architecture.contracts import AgentAnalysisResult
from src.core.models import QuarterlyFinancials, SymbolMetadata
from src.core.types import AgentStatus, SignalType


def _q(symbol: str, available: date, pat_growth: float, period_end: date) -> QuarterlyFinancials:
    return QuarterlyFinancials(
        symbol=symbol,
        period_end_date=period_end,
        filing_date=available,
        available_at=available,
        sales_crores=1000.0,
        sales_growth_yoy_pct=15.0,
        pat_crores=150.0,
        pat_growth_yoy_pct=pat_growth,
        ebitda_margin_pct=18.0,
        eps_inr=12.0,
        pit_status="VERIFIED",
    )


def test_fundamental_agent_emits_common_contract():
    agent = FundamentalAnalysisAgent()
    result = asyncio.run(agent.analyze_contract(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail"),
        pd.DataFrame(),
        date(2026, 5, 10),
        context={"quarterly_financials": [_q("TRENT", date(2026, 2, 10), 20.0, date(2025, 12, 31))]},
    ))

    assert isinstance(result, AgentAnalysisResult)
    assert result.symbol == "TRENT"
    assert result.decision_time == date(2026, 5, 10)
    assert result.pit_safe is True
    assert result.status == AgentStatus.SUCCESS
    assert result.signal in {SignalType.BULLISH, SignalType.BEARISH, SignalType.NEUTRAL}
    assert all(e.source == "FUNDAMENTAL" and e.pit_safe for e in result.evidence)


def test_future_fundamental_mutation_cannot_change_result_at_T():
    agent = FundamentalAnalysisAgent()
    t = date(2026, 5, 10)
    baseline = _q("TRENT", date(2026, 2, 10), 20.0, date(2025, 12, 31))
    future_good = _q("TRENT", date(2026, 5, 25), 60.0, date(2026, 3, 31))
    future_bad = _q("TRENT", date(2026, 5, 25), -90.0, date(2026, 3, 31))
    meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")

    base = asyncio.run(agent.analyze_contract(meta, pd.DataFrame(), t,
        context={"quarterly_financials": [baseline, future_good]}))
    mutated = asyncio.run(agent.analyze_contract(meta, pd.DataFrame(), t,
        context={"quarterly_financials": [baseline, future_bad]}))

    assert base.score == mutated.score
    assert base.signal == mutated.signal
    assert base.pit_safe == mutated.pit_safe
    assert [e.observation for e in base.evidence] == [e.observation for e in mutated.evidence]


def test_future_only_fundamentals_return_unknown_not_neutral():
    agent = FundamentalAnalysisAgent()
    result = asyncio.run(agent.analyze_contract(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail"),
        pd.DataFrame(),
        date(2026, 5, 10),
        context={"quarterly_financials": [_q("TRENT", date(2026, 5, 25), 40.0, date(2026, 3, 31))]},
    ))

    assert result.status == AgentStatus.DATA_UNAVAILABLE
    assert result.signal == SignalType.UNKNOWN
    assert result.score == 0.0
    assert result.pit_safe is False
    assert result.reasons


def test_missing_fundamentals_return_unknown():
    agent = FundamentalAnalysisAgent()
    result = asyncio.run(agent.analyze_contract(
        SymbolMetadata(symbol="UNKNOWN", company_name="Unknown Co"),
        pd.DataFrame(),
        date(2026, 5, 10),
        context={},
    ))

    assert result.status == AgentStatus.DATA_UNAVAILABLE
    assert result.signal == SignalType.UNKNOWN
    assert result.pit_safe is False
    assert result.score == 0.0


def test_latest_visible_quarterly_record_drives_evidence():
    agent = FundamentalAnalysisAgent()
    q_old = _q("TRENT", date(2026, 2, 10), 15.0, date(2025, 12, 31))
    q_new = _q("TRENT", date(2026, 5, 5), 30.0, date(2026, 3, 31))
    result = asyncio.run(agent.analyze_contract(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail"),
        pd.DataFrame(),
        date(2026, 5, 10),
        context={"quarterly_financials": [q_old, q_new]},
    ))

    assert result.pit_safe is True
    assert any("30.0%" in e.observation for e in result.evidence)
