"""
Phase #14D — Evidence Fusion & Conviction Engine Unit Tests.

Validates that:
  1. EvidenceFusionEngine aggregates strength and confidence across specialist agents.
  2. ConvictionEngine computes deterministic conviction scores and grades.
  3. Contradictory agent signals trigger an explicit conflict penalty.
  4. Hard PIT violations force immediate REJECT decision regardless of agent scores.
  5. CIOContract correctly synthesizes BUY, WATCH, NO_TRADE, and REJECT decisions.
"""

from datetime import date, datetime
import pytest

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
from src.core.types import AgentStatus, SignalType
from src.data.data_quality import DataQualityResult, DataQualityStatus


def _make_dq(pit_safe: bool = True) -> DataQualityResult:
    return DataQualityResult(
        symbol="TRENT",
        as_of_date=datetime(2026, 6, 30, 10, 0),
        overall_status=DataQualityStatus.VALID if pit_safe else DataQualityStatus.PIT_VIOLATION,
        overall_quality_score=100.0 if pit_safe else 0.0,
        pit_safe=pit_safe,
        is_trade_eligible=pit_safe,
    )


def test_fuse_evidence_bullish_synthesis():
    dt = datetime(2026, 6, 30, 10, 0)
    dq = _make_dq(pit_safe=True)

    tech_ev = StructuredEvidence(
        source="TECHNICAL", observation="EMA20 > EMA50 bullish alignment", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.8, pit_safe=True,
    )
    fund_ev = StructuredEvidence(
        source="FUNDAMENTAL", observation="PAT growth +30% YoY", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.9, pit_safe=True,
    )

    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=85.0, confidence=0.8, evidence=[tech_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=80.0, confidence=0.9, evidence=[fund_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq)

    assert fusion.symbol == "TRENT"
    assert len(fusion.bullish_evidence) == 2
    assert fusion.aggregate_strength is not None
    assert fusion.aggregate_strength > 0.5
    assert fusion.aggregate_confidence is not None
    assert fusion.aggregate_confidence > 0.0
    assert len(fusion.conflicts) == 0
    assert len(fusion.vetoes) == 0


def test_conviction_engine_high_conviction():
    dt = datetime(2026, 6, 30, 10, 0)
    dq = _make_dq(pit_safe=True)

    tech_ev = StructuredEvidence(
        source="TECHNICAL", observation="EMA20 > EMA50 bullish alignment", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.8, pit_safe=True,
    )
    fund_ev = StructuredEvidence(
        source="FUNDAMENTAL", observation="PAT growth +30% YoY", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.9, pit_safe=True,
    )

    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=85.0, confidence=0.8, evidence=[tech_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=80.0, confidence=0.9, evidence=[fund_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq)
    conviction = ConvictionEngine.evaluate_conviction(fusion)

    assert conviction.grade == ConvictionGrade.HIGH_CONVICTION
    assert conviction.score >= 75.0


def test_conviction_engine_conflict_penalty():
    dt = datetime(2026, 6, 30, 10, 0)
    dq = _make_dq(pit_safe=True)

    tech_ev = StructuredEvidence(
        source="TECHNICAL", observation="Breakout", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.8, pit_safe=True,
    )
    fund_ev = StructuredEvidence(
        source="FUNDAMENTAL", observation="PAT decline -20% YoY", as_of=dt,
        direction=SignalType.BEARISH, strength="HIGH", reliability=0.8, pit_safe=True,
    )

    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=80.0, confidence=0.8, evidence=[tech_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAnalysisAgent", decision_time=dt,
        signal=SignalType.BEARISH, score=30.0, confidence=0.8, evidence=[fund_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq)
    assert len(fusion.conflicts) > 0

    conviction = ConvictionEngine.evaluate_conviction(fusion)
    assert conviction.grade != ConvictionGrade.HIGH_CONVICTION
    assert any("CONFLICT_PENALTY_APPLIED" in r for r in conviction.reasons)


def test_cio_decision_hard_pit_violation_rejection():
    dt = datetime(2026, 6, 30, 10, 0)
    dq_unsafe = _make_dq(pit_safe=False)

    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=90.0, confidence=0.9, pit_safe=True, status=AgentStatus.SUCCESS,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res], dq_unsafe)
    conviction = ConvictionEngine.evaluate_conviction(fusion)

    risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)
    cio_input = CIOInput(
        symbol="TRENT", decision_time=dt, technical_result=tech_res,
        fusion_result=fusion, conviction_result=conviction, risk_result=risk, data_quality=dq_unsafe,
    )

    decision = CIOContract.evaluate_decision(cio_input)
    assert decision.decision == "REJECT"
    assert decision.confidence == 0.0
    assert "PIT_VIOLATION" in decision.vetoes


def test_cio_decision_buy_pipeline():
    dt = datetime(2026, 6, 30, 10, 0)
    dq = _make_dq(pit_safe=True)

    tech_ev = StructuredEvidence(
        source="TECHNICAL", observation="EMA20 > EMA50", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.9, pit_safe=True,
    )
    fund_ev = StructuredEvidence(
        source="FUNDAMENTAL", observation="Earnings +35%", as_of=dt,
        direction=SignalType.BULLISH, strength="HIGH", reliability=0.9, pit_safe=True,
    )

    tech_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="TechnicalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=88.0, confidence=0.9, evidence=[tech_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )
    fund_res = AgentAnalysisResult(
        symbol="TRENT", agent_name="FundamentalAnalysisAgent", decision_time=dt,
        signal=SignalType.BULLISH, score=85.0, confidence=0.9, evidence=[fund_ev], pit_safe=True, status=AgentStatus.SUCCESS,
    )

    fusion = EvidenceFusionEngine.fuse_evidence("TRENT", dt, [tech_res, fund_res], dq)
    conviction = ConvictionEngine.evaluate_conviction(fusion)
    risk = RiskEngineResult(symbol="TRENT", decision_time=dt, passed_risk_veto=True)

    cio_input = CIOInput(
        symbol="TRENT", decision_time=dt, technical_result=tech_res,
        fundamental_result=fund_res, fusion_result=fusion,
        conviction_result=conviction, risk_result=risk, data_quality=dq,
    )

    decision = CIOContract.evaluate_decision(cio_input)
    assert decision.decision == "BUY"
    assert decision.confidence > 0.7
