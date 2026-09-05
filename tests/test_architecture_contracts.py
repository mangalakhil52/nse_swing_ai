"""
P0 #14A CORRECTION — Multi-Agent Architecture & Structured Contracts Unit Tests.

Validates that:
  1. EvidenceFusionEngine does NOT compute a magic average (aggregate_strength is None, aggregate_confidence is None).
  2. ConvictionEngine does NOT apply arbitrary strategy thresholds in #14A (returns ConvictionGrade.NOT_COMPUTED).
  3. Agent PIT veto (pit_safe = False) is recorded in fusion vetoes and reaches CIO (forcing decision = REJECT).
  4. SignalType.UNKNOWN and SignalType.NEUTRAL remain separately represented (unknown_evidence vs neutral_evidence).
  5. Opposing BULLISH vs BEARISH views preserve agent disagreement in conflicts without magic score averaging.
  6. Unsupplied PIT metadata is NOT implicitly trusted (defaults pit_safe = False).
  7. All agents share the common AgentAnalysisResult contract with explicit decision_time.
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


def test_fusion_does_not_compute_magic_average():
    """1. Test EvidenceFusionEngine does NOT calculate a magic average score."""
    dt = datetime(2026, 8, 19, 10, 0)
    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=90.0, confidence=0.8, pit_safe=True,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=60.0, confidence=0.6, pit_safe=True,
    )

    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq_res)

    assert fusion.aggregate_strength is None
    assert fusion.aggregate_confidence is None


def test_conviction_does_not_apply_arbitrary_thresholds():
    """2. Test ConvictionEngine returns LOW_CONVICTION when evidence is empty."""
    dt = datetime(2026, 8, 19, 10, 0)
    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = FusionResult(symbol="TRENT", decision_time=dt, data_quality=dq_res)
    conviction = ConvictionEngine.evaluate_conviction(fusion)

    assert conviction.grade == ConvictionGrade.NOT_COMPUTED
    assert any("INSUFFICIENT" in r for r in conviction.reasons)


def test_agent_pit_veto_reaches_cio():
    """3. Test agent with pit_safe=False generates PIT veto in fusion and reaches CIO (forcing REJECT)."""
    dt = datetime(2026, 8, 19, 10, 0)
    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=95.0, confidence=0.9, pit_safe=False,
    )

    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res], dq_res)

    assert "AGENT_PIT_VIOLATION_TECHNICALAGENT" in fusion.vetoes
    assert "PIT_VIOLATION" in fusion.vetoes

    conviction = ConvictionResult(symbol="TRENT", decision_time=dt, grade=ConvictionGrade.HIGH_CONVICTION, score=95.0)
    risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)

    cio_input = CIOInput(
        symbol="TRENT",
        decision_time=dt,
        technical_result=tech_res,
        fusion_result=fusion,
        conviction_result=conviction,
        risk_result=risk,
        data_quality=dq_res,
    )

    cio_decision = CIOContract.evaluate_decision(cio_input)

    assert cio_decision.decision == "REJECT"
    assert cio_decision.decision != "BUY"
    assert "PIT_VIOLATION" in cio_decision.vetoes


def test_unknown_is_distinct_from_neutral():
    """4. Test SignalType.UNKNOWN and SignalType.NEUTRAL remain separately represented."""
    dt = datetime(2026, 8, 19, 10, 0)
    ev_neutral = StructuredEvidence(
        source="TECHNICAL", observation="RSI in neutral range (50)", as_of=dt,
        direction=SignalType.NEUTRAL, strength="MEDIUM", pit_safe=True,
    )
    ev_unknown = StructuredEvidence(
        source="NEWS", observation="No news articles found", as_of=dt,
        direction=SignalType.UNKNOWN, strength="LOW", pit_safe=True,
    )

    agent_neutral = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=dt,
        signal=SignalType.NEUTRAL, evidence=[ev_neutral], pit_safe=True,
    )
    agent_unknown = AgentAnalysisResult(
        symbol="TRENT", agent_name="NewsAgent", decision_time=dt,
        signal=SignalType.UNKNOWN, evidence=[ev_unknown], pit_safe=True,
    )

    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [agent_neutral, agent_unknown], dq_res)

    assert len(fusion.neutral_evidence) == 1
    assert len(fusion.unknown_evidence) == 1
    assert fusion.neutral_evidence[0].direction == SignalType.NEUTRAL
    assert fusion.unknown_evidence[0].direction == SignalType.UNKNOWN


def test_conflicting_agents_are_not_averaged():
    """5. Test conflicting BULLISH and BEARISH views are preserved in conflicts without score averaging."""
    dt = datetime(2026, 8, 19, 10, 0)
    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=95.0, confidence=0.9, pit_safe=True,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAgent", decision_time=dt,
        signal=SignalType.BEARISH, score=20.0, confidence=0.8, pit_safe=True,
    )

    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq_res)

    assert len(fusion.conflicts) > 0
    assert "CONTRADICTORY_SIGNALS" in fusion.conflicts[0]
    assert fusion.aggregate_strength is None
    assert fusion.aggregate_confidence is None


def test_missing_pit_metadata_is_not_implicitly_trusted():
    """6. Test an agent result created without explicit pit_safe defaults to pit_safe = False."""
    dt = datetime(2026, 8, 19, 10, 0)
    agent_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="UnverifiedAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=80.0,
        # pit_safe NOT explicitly set to True
    )

    assert agent_res.pit_safe is False

    dq_res = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [agent_res], dq_res)

    assert "AGENT_PIT_VIOLATION_UNVERIFIEDAGENT" in fusion.vetoes
    assert "PIT_VIOLATION" in fusion.vetoes


def test_all_agents_share_common_result_contract():
    """7. Test AgentAnalysisResult provides all required common fields."""
    now = datetime(2026, 8, 19, 10, 0)
    ev = StructuredEvidence(
        source="TECHNICAL", observation="Breakout above 20 EMA", as_of=now,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.9, pit_safe=True,
    )

    res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAgent", decision_time=now,
        signal=SignalType.BULLISH, score=85.0, confidence=0.8,
        evidence=[ev], risks=["High Volatility"], pit_safe=True, status=AgentStatus.SUCCESS,
    )

    assert res.symbol == "TRENT"
    assert res.agent_name == "TechnicalAgent"
    assert res.decision_time == now
    assert res.signal == SignalType.BULLISH
    assert res.score == 85.0
    assert res.confidence == 0.8
    assert len(res.evidence) == 1
    assert res.pit_safe is True


def test_agents_cannot_directly_construct_portfolio_trades():
    """8. Test CIO returns decision strings (BUY, WATCH, NO_TRADE, REJECT), NOT trade/portfolio execution objects."""
    dt = datetime(2026, 8, 19, 10, 0)
    dq_valid = DataQualityResult(
        symbol="TRENT", as_of_date=dt, overall_status=DataQualityStatus.VALID,
        overall_quality_score=100.0, pit_safe=True, is_trade_eligible=True,
    )

    fusion = FusionResult(symbol="TRENT", decision_time=dt, data_quality=dq_valid)
    conviction = ConvictionResult(symbol="TRENT", decision_time=dt, grade=ConvictionGrade.HIGH_CONVICTION, score=85.0)
    risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)

    cio_input = CIOInput(
        symbol="TRENT", decision_time=dt, fusion_result=fusion,
        conviction_result=conviction, risk_result=risk, data_quality=dq_valid,
    )

    decision = CIOContract.evaluate_decision(cio_input)

    assert decision.decision == "BUY"
    assert isinstance(decision.decision, str)
    assert not hasattr(decision, "shares")  # No portfolio/execution object
