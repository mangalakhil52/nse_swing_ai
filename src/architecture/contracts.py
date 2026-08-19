"""
Multi-Agent Architecture & Structured Contracts Module — src/architecture/contracts.py (P0 #14A)

Defines production data contracts, structured evidence models, agent boundaries,
and deterministic synthesis contracts for:
  - AgentAnalysisResult
  - StructuredEvidence
  - EvidenceFusionEngine & FusionResult
  - ConvictionEngine & ConvictionResult
  - RiskEngine & RiskEngineResult
  - CIOContract, CIOInput & CIODecision

Enforces:
  1. Agents produce structured evidence; they DO NOT construct or execute trades directly.
  2. Decision time is explicit (backtesting & live shared semantics).
  3. Signal, score, and confidence remain strictly separate.
  4. Missing evidence produces SignalType.UNKNOWN (NOT NEUTRAL or BULLISH).
  5. CIO CANNOT override hard PIT violations or data quality vetoes.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from src.core.types import AgentStatus, SignalType
from src.data.data_quality import DataQualityResult, SourceQualityResult, DataQualityStatus


class VetoType(str, Enum):
    DATA_QUALITY_VETO = "DATA_QUALITY_VETO"
    PIT_VIOLATION = "PIT_VIOLATION"
    RISK_VETO = "RISK_VETO"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REGIME_VETO = "REGIME_VETO"
    PORTFOLIO_CAPITAL_VETO = "PORTFOLIO_CAPITAL_VETO"


class ConvictionGrade(str, Enum):
    HIGH_CONVICTION = "HIGH_CONVICTION"
    MEDIUM_CONVICTION = "MEDIUM_CONVICTION"
    LOW_CONVICTION = "LOW_CONVICTION"
    REJECTED = "REJECTED"


class StructuredEvidence(BaseModel):
    """Structured, explainable evidence item emitted by specialist research agents."""
    source: str = Field(..., description="e.g. 'TECHNICAL', 'FUNDAMENTAL', 'NEWS', 'MARKET_REGIME'")
    observation: str
    as_of: datetime | date
    direction: SignalType
    strength: str = Field(default="MEDIUM", description="'HIGH', 'MEDIUM', 'LOW'")
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    pit_safe: bool = Field(default=True)


class AgentAnalysisResult(BaseModel):
    """Common output contract returned by all domain-specific research agents."""
    symbol: str
    agent_name: str
    decision_time: datetime | date
    signal: SignalType = Field(default=SignalType.UNKNOWN)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[StructuredEvidence] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    data_quality: SourceQualityResult | DataQualityResult | None = None
    pit_safe: bool = Field(default=True)
    status: AgentStatus = Field(default=AgentStatus.SUCCESS)
    reasons: list[str] = Field(default_factory=list)


class FusionResult(BaseModel):
    """Result of fusing structured evidence from multiple specialist agents."""
    symbol: str
    decision_time: datetime | date
    bullish_evidence: list[StructuredEvidence] = Field(default_factory=list)
    bearish_evidence: list[StructuredEvidence] = Field(default_factory=list)
    unknown_evidence: list[StructuredEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    aggregate_strength: float = Field(default=0.0, ge=0.0, le=100.0)
    aggregate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_quality: DataQualityResult
    vetoes: list[str] = Field(default_factory=list)


class ConvictionResult(BaseModel):
    """Evaluation of evidence conviction strength prior to risk/CIO synthesis."""
    symbol: str
    decision_time: datetime | date
    grade: ConvictionGrade
    score: float = Field(..., ge=0.0, le=100.0)
    reasons: list[str] = Field(default_factory=list)


class RiskEngineResult(BaseModel):
    """Independent risk evaluation result that can veto trade consideration."""
    symbol: str
    decision_time: datetime | date
    passed_risk_veto: bool
    veto_category: VetoType | None = None
    max_position_size_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    stop_loss_valid: bool = Field(default=True)
    reasons: list[str] = Field(default_factory=list)


class CIOInput(BaseModel):
    """Structured input package delivered to CIO for final synthesis."""
    symbol: str
    decision_time: datetime | date
    technical_result: AgentAnalysisResult | None = None
    fundamental_result: AgentAnalysisResult | None = None
    news_result: AgentAnalysisResult | None = None
    regime_result: AgentAnalysisResult | None = None
    fusion_result: FusionResult
    conviction_result: ConvictionResult
    risk_result: RiskEngineResult
    data_quality: DataQualityResult


class CIODecision(BaseModel):
    """Final synthesis decision emitted by the Chief Investment Officer."""
    symbol: str
    decision_time: datetime | date
    decision: str = Field(..., description="'BUY', 'WATCH', 'NO_TRADE', 'REJECT'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    vetoes: list[str] = Field(default_factory=list)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)


class EvidenceFusionEngine:
    """Combines structured agent evidence without hardcoding magic average weights."""

    @classmethod
    def fuse_evidence(
        cls,
        symbol: str,
        decision_time: datetime | date,
        agent_results: list[AgentAnalysisResult],
        data_quality: DataQualityResult,
    ) -> FusionResult:
        bullish: list[StructuredEvidence] = []
        bearish: list[StructuredEvidence] = []
        unknown: list[StructuredEvidence] = []
        conflicts: list[str] = []
        vetoes: list[str] = []

        # Collect evidence across all agent results
        for agent_res in agent_results:
            if not agent_res.pit_safe:
                vetoes.append(f"AGENT_PIT_VIOLATION_{agent_res.agent_name.upper()}")

            if agent_res.signal == SignalType.BULLISH:
                bullish.extend(agent_res.evidence)
            elif agent_res.signal == SignalType.BEARISH:
                bearish.extend(agent_res.evidence)
            elif agent_res.signal == SignalType.UNKNOWN or agent_res.signal == SignalType.NEUTRAL:
                unknown.extend(agent_res.evidence)

        # Detect agent disagreements
        signals = {res.agent_name: res.signal for res in agent_results}
        if SignalType.BULLISH in signals.values() and SignalType.BEARISH in signals.values():
            conflicts.append("CONTRADICTORY_SIGNALS: Agents express opposing BULLISH and BEARISH views.")

        # Data quality hard vetoes
        if not data_quality.pit_safe or data_quality.overall_status == DataQualityStatus.PIT_VIOLATION:
            vetoes.append("DATA_QUALITY_PIT_VIOLATION")

        valid_scores = [r.score for r in agent_results if r.signal not in (SignalType.UNKNOWN, SignalType.REJECT)]
        agg_strength = round(sum(valid_scores) / max(len(valid_scores), 1), 1) if valid_scores else 0.0

        valid_confs = [r.confidence for r in agent_results if r.signal != SignalType.UNKNOWN]
        agg_conf = round(sum(valid_confs) / max(len(valid_confs), 1), 2) if valid_confs else 0.0

        return FusionResult(
            symbol=symbol,
            decision_time=decision_time,
            bullish_evidence=bullish,
            bearish_evidence=bearish,
            unknown_evidence=unknown,
            conflicts=conflicts,
            aggregate_strength=agg_strength,
            aggregate_confidence=agg_conf,
            data_quality=data_quality,
            vetoes=vetoes,
        )


class ConvictionEngine:
    """Evaluates evidence strength prior to risk/CIO synthesis."""

    @classmethod
    def evaluate_conviction(cls, fusion_result: FusionResult) -> ConvictionResult:
        if fusion_result.vetoes or not fusion_result.data_quality.pit_safe:
            return ConvictionResult(
                symbol=fusion_result.symbol,
                decision_time=fusion_result.decision_time,
                grade=ConvictionGrade.REJECTED,
                score=0.0,
                reasons=["REJECTED_DUE_TO_VETOES_OR_PIT_FAILURE"],
            )

        if fusion_result.aggregate_strength >= 80.0 and fusion_result.aggregate_confidence >= 0.7:
            grade = ConvictionGrade.HIGH_CONVICTION
        elif fusion_result.aggregate_strength >= 60.0 and fusion_result.aggregate_confidence >= 0.5:
            grade = ConvictionGrade.MEDIUM_CONVICTION
        elif fusion_result.aggregate_strength >= 40.0:
            grade = ConvictionGrade.LOW_CONVICTION
        else:
            grade = ConvictionGrade.REJECTED

        return ConvictionResult(
            symbol=fusion_result.symbol,
            decision_time=fusion_result.decision_time,
            grade=grade,
            score=fusion_result.aggregate_strength,
            reasons=[f"Grade assigned: {grade}"],
        )


class CIOContract:
    """Synthesizes structured inputs into final decision (BUY, WATCH, NO_TRADE, REJECT)."""

    @classmethod
    def evaluate_decision(cls, cio_input: CIOInput) -> CIODecision:
        # Inviolable Rule: CIO CANNOT override hard PIT violations or data quality vetoes!
        if not cio_input.data_quality.pit_safe or cio_input.data_quality.overall_status == DataQualityStatus.PIT_VIOLATION:
            return CIODecision(
                symbol=cio_input.symbol,
                decision_time=cio_input.decision_time,
                decision="REJECT",
                confidence=0.0,
                reasons=["PIT_VIOLATION: Hard failure detected in data quality."],
                vetoes=["PIT_VIOLATION"],
            )

        if not cio_input.risk_result.passed_risk_veto:
            return CIODecision(
                symbol=cio_input.symbol,
                decision_time=cio_input.decision_time,
                decision="NO_TRADE",
                confidence=0.0,
                reasons=cio_input.risk_result.reasons,
                vetoes=[str(cio_input.risk_result.veto_category or "RISK_VETO")],
            )

        if cio_input.conviction_result.grade == ConvictionGrade.HIGH_CONVICTION:
            decision = "BUY"
        elif cio_input.conviction_result.grade in (ConvictionGrade.MEDIUM_CONVICTION, ConvictionGrade.LOW_CONVICTION):
            decision = "WATCH"
        else:
            decision = "NO_TRADE"

        return CIODecision(
            symbol=cio_input.symbol,
            decision_time=cio_input.decision_time,
            decision=decision,
            confidence=cio_input.fusion_result.aggregate_confidence,
            reasons=cio_input.conviction_result.reasons,
            vetoes=cio_input.fusion_result.vetoes,
        )
