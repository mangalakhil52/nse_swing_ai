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
    NOT_COMPUTED = "NOT_COMPUTED"


class StructuredEvidence(BaseModel):
    """Structured, explainable evidence item emitted by specialist research agents."""
    source: str = Field(..., description="e.g. 'TECHNICAL', 'FUNDAMENTAL', 'NEWS', 'MARKET_REGIME'")
    observation: str
    as_of: datetime | date
    direction: SignalType
    strength: str = Field(default="MEDIUM", description="'HIGH', 'MEDIUM', 'LOW'")
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    pit_safe: bool = Field(default=False)


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
    pit_safe: bool = Field(default=False)  # Unsupplied metadata is NOT implicitly trusted
    status: AgentStatus = Field(default=AgentStatus.SUCCESS)
    reasons: list[str] = Field(default_factory=list)


class FusionResult(BaseModel):
    """Result of fusing structured evidence from multiple specialist agents."""
    symbol: str
    decision_time: datetime | date
    bullish_evidence: list[StructuredEvidence] = Field(default_factory=list)
    bearish_evidence: list[StructuredEvidence] = Field(default_factory=list)
    neutral_evidence: list[StructuredEvidence] = Field(default_factory=list)
    unknown_evidence: list[StructuredEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    aggregate_strength: float | None = Field(default=None)
    aggregate_confidence: float | None = Field(default=None)
    data_quality: DataQualityResult
    vetoes: list[str] = Field(default_factory=list)


class ConvictionResult(BaseModel):
    """Evaluation of evidence conviction strength prior to risk/CIO synthesis."""
    symbol: str
    decision_time: datetime | date
    grade: ConvictionGrade
    score: float = Field(default=0.0, ge=0.0, le=100.0)
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
    """Combines structured agent evidence without magic average weights or fake scoring models."""

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
        neutral: list[StructuredEvidence] = []
        unknown: list[StructuredEvidence] = []
        conflicts: list[str] = []
        vetoes: list[str] = []

        # Collect evidence across all agent results without collapsing UNKNOWN and NEUTRAL
        for agent_res in agent_results:
            if not agent_res.pit_safe:
                vetoes.append(f"AGENT_PIT_VIOLATION_{agent_res.agent_name.upper()}")
                vetoes.append("PIT_VIOLATION")

            if agent_res.signal == SignalType.BULLISH:
                bullish.extend(agent_res.evidence)
            elif agent_res.signal == SignalType.BEARISH:
                bearish.extend(agent_res.evidence)
            elif agent_res.signal == SignalType.NEUTRAL:
                neutral.extend(agent_res.evidence)
            elif agent_res.signal == SignalType.UNKNOWN:
                unknown.extend(agent_res.evidence)

        # Detect agent disagreements and preserve them explicitly
        signals = {res.agent_name: res.signal for res in agent_results}
        if SignalType.BULLISH in signals.values() and SignalType.BEARISH in signals.values():
            conflicts.append("CONTRADICTORY_SIGNALS: Agents express opposing BULLISH and BEARISH views.")

        # Data quality hard vetoes
        if not data_quality.pit_safe or data_quality.overall_status == DataQualityStatus.PIT_VIOLATION:
            vetoes.append("DATA_QUALITY_PIT_VIOLATION")
            vetoes.append("PIT_VIOLATION")

        # Invariant: NO magic score averaging inside #14A
        return FusionResult(
            symbol=symbol,
            decision_time=decision_time,
            bullish_evidence=bullish,
            bearish_evidence=bearish,
            neutral_evidence=neutral,
            unknown_evidence=unknown,
            conflicts=conflicts,
            aggregate_strength=None,  # Not computed in #14A
            aggregate_confidence=None,  # Not computed in #14A
            data_quality=data_quality,
            vetoes=list(set(vetoes)),
        )


class ConvictionEngine:
    """Evaluates evidence conviction contract without arbitrary strategy thresholds."""

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

        # Contract placeholder: Conviction methodology is not computed in #14A
        return ConvictionResult(
            symbol=fusion_result.symbol,
            decision_time=fusion_result.decision_time,
            grade=ConvictionGrade.NOT_COMPUTED,
            score=0.0,
            reasons=["CONVICTION_METHODOLOGY_NOT_COMPUTED_IN_14A"],
        )


class CIOContract:
    """Synthesizes structured inputs into final decision (BUY, WATCH, NO_TRADE, REJECT)."""

    @classmethod
    def evaluate_decision(cls, cio_input: CIOInput) -> CIODecision:
        # Inviolable Rule: CIO CANNOT override hard PIT violations or data quality vetoes!
        hard_pit_vetoes = [
            v for v in cio_input.fusion_result.vetoes
            if v == "PIT_VIOLATION" or v == "DATA_QUALITY_PIT_VIOLATION" or v.startswith("AGENT_PIT_VIOLATION_")
        ]

        if not cio_input.data_quality.pit_safe or cio_input.data_quality.overall_status == DataQualityStatus.PIT_VIOLATION or hard_pit_vetoes:
            all_vetoes = list(set(cio_input.fusion_result.vetoes + ["PIT_VIOLATION"]))
            return CIODecision(
                symbol=cio_input.symbol,
                decision_time=cio_input.decision_time,
                decision="REJECT",
                confidence=0.0,
                reasons=["HARD_PIT_VIOLATION: Cannot construct BUY decision due to hard PIT failure."],
                vetoes=all_vetoes,
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
            confidence=0.0,
            reasons=cio_input.conviction_result.reasons,
            vetoes=cio_input.fusion_result.vetoes,
        )
