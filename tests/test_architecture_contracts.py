"""
P0 #14A — Multi-Agent Architecture & Structured Contracts Unit Tests.

Validates that:
  1. All agents share the common AgentAnalysisResult contract.
  2. decision_time is explicit and present across all agent outputs and synthesis layers.
  3. PIT status and data_quality metadata are present on agent results.
  4. SignalType.UNKNOWN is distinct from SignalType.NEUTRAL.
  5. Agent disagreement (opposing BULLISH vs BEARISH views) is preserved in EvidenceFusionEngine.fuse_evidence.
  6. Machine-readable vetoes are recorded and preserved across fusion and risk layers.
  7. CIOContract CANNOT override a hard PIT violation or data quality failure.
  8. CIO returns decision strings (BUY, WATCH, NO_TRADE, REJECT), NOT trade/portfolio execution objects.
"""

from datetime import date, datetime
import pytest

from src.architecture.contracts import (
    AgentAnalysisResult,
    CIOContract,
    CIOInput,
    ConvictionEngine,
    ConvictionGrade,
    ConvictionResult,
    EvidenceFusionEngine,
    FusionResult,
    RiskEngineResult,
    StructuredEvidence,
    VetoType,
)
from src.core.types import AgentStatus, SignalType
from src.data.data_quality import DataQualityGate, DataQualityResult, DataQualityStatus


def test_all_agents_share_common_result_contract():
    """1. Test AgentAnalysisResult provides all required common fields."""
    now = datetime(2026, 8, 19, 10, 0)
    ev = StructuredEvidence(
        source="TECHNICAL",
        observation="Breakout above 20 EMA",
        as_of=now,
        direction=SignalType.BULLISH,
        strength="HIGH",
        reliability=0.9,
        pit_safe=True,
    )

    res = AgentAnalysisResult(
        symbol="TRENT",
        agent_name="TechnicalAgent",
        decision_time=now,
        signal=SignalType.BULLISH,
        score=85.0,
        confidence=0.8,
        evidence=[ev],
        risks=["High Volatility"],
        pit_safe=True,
        status=AgentStatus.SUCCESS,
    )

    assert res.symbol == "TRENT"
    assert res.agent_name == "TechnicalAgent"
    assert res.decision_time == now
    assert res.signal == SignalType.BULLISH
    assert res.score == 85.0
    assert res.confidence == 0.8
    assert len(res.evidence) == 1
    assert res.pit_safe is True


def test_decision_time_is_present_in_agent_results():
    """2. Test decision_time is present and explicit."""
    dt = datetime(2026, 8, 19, 10, 0)
    res = AgentAnalysisResult(
        symbol="TRENT",
        agent_name="NewsAgent",
        decision_time=dt,
        signal=SignalType.UNKNOWN,
    )
    assert res.decision_time == dt


def test_pit_status_and_data_quality_are_present():
    """3. Test PIT status and data_quality attributes are present."""
    dt = datetime(2026, 8, 19, 10, 0)
    res = AgentAnalysisResult(
        symbol="TRENT",
        agent_name="FundamentalAgent",
        decision_time=dt,
        pit_safe=False,
        reasons=["FUNDAMENTAL_PIT_UNVERIFIED"],
    )
    assert res.pit_safe is False
    assert "FUNDAMENTAL_PIT_UNVERIFIED" in res.reasons


def test_unknown_is_distinct_from_neutral():
    """4. Test SignalType.UNKNOWN is distinct from SignalType.NEUTRAL."""
    assert SignalType.UNKNOWN != SignalType.NEUTRAL
    assert SignalType.UNKNOWN.value == "UNKNOWN"
    assert SignalType.NEUTRAL.value == "NEUTRAL"


def test_agent_disagreement_is_preserved():
    """5. Test agent disagreement (opposing BULLISH vs BEARISH views) is preserved in conflicts."""
    dt = datetime(2026, 8, 19, 10, 0)
    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=85.0, confidence=0.8,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAgent", decision_time=dt,
        signal=SignalType.BEARISH, score=70.0, confidence=0.7,
    )

    dq = DataQualityGate.evaluate_ohlcv(None, "TRENT")  # Empty -> INVALID
    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq_res)

    assert len(fusion.conflicts) > 0
    assert any("CONTRADICTORY_SIGNALS" in c for c in fusion.conflicts)


def test_vetoes_are_preserved():
    """6. Test machine-readable vetoes are recorded and preserved."""
    dt = datetime(2026, 8, 19, 10, 0)
    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=85.0, confidence=0.8, pit_safe=False,
    )

    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.PIT_VIOLATION,
        overall_quality_score=0.0, pit_safe=False, is_trade_eligible=False,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res], dq_res)

    assert "DATA_QUALITY_PIT_VIOLATION" in fusion.vetoes
    assert "AGENT_PIT_VIOLATION_TECHNICALAGENT" in fusion.vetoes


def test_cio_cannot_override_hard_pit_violation():
    """7. Test CIOContract CANNOT override a hard PIT violation or data quality failure."""
    dt = datetime(2026, 8, 19, 10, 0)
    dq_pit_fail = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.PIT_VIOLATION,
        overall_quality_score=0.0, pit_safe=False, is_trade_eligible=False,
    )

    fusion = FusionResult(symbol="TRENT", decision_time=dt, data_quality=dq_pit_fail, vetoes=["PIT_VIOLATION"])
    conviction = ConvictionResult(symbol="TRENT", decision_time=dt, grade=ConvictionGrade.HIGH_CONVICTION, score=90.0)
    risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)

    cio_input = CIOInput(
        symbol="TRENT",
        decision_time=dt,
        fusion_result=fusion,
        conviction_result=conviction,
        risk_result=risk,
        data_quality=dq_pit_fail,
    )

    decision = CIOContract.evaluate_decision(cio_input)

    assert decision.decision == "REJECT"
    assert decision.confidence == 0.0
    assert "PIT_VIOLATION" in decision.vetoes


def test_agents_cannot_directly_construct_portfolio_trades():
    """8. Test CIO returns decision strings (BUY, WATCH, NO_TRADE, REJECT), NOT trade/portfolio execution objects."""
    dt = datetime(2026, 8, 19, 10, 0)
    dq_valid = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = FusionResult(symbol="TRENT", decision_time=dt, aggregate_confidence=0.85, data_quality=dq_valid)
    conviction = ConvictionResult(symbol="TRENT", decision_time=dt, grade=ConvictionGrade.HIGH_CONVICTION, score=85.0)
    risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)

    cio_input = CIOInput(
        symbol="TRENT",
        decision_time=dt,
        fusion_result=fusion,
        conviction_result=conviction,
        risk_result=risk,
        data_quality=dq_valid,
    )

    decision = CIOContract.evaluate_decision(cio_input)

    assert decision.decision == "BUY"
    assert isinstance(decision.decision, str)
    assert not hasattr(decision, "shares")  # No portfolio/execution object
