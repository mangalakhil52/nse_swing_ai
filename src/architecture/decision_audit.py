"""#14K deterministic, reproducible decision audit record."""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class DecisionAuditRecord:
    symbol: str
    decision_time: date | datetime
    final_action: str
    final_score: float
    final_confidence: float
    agent_statuses: dict[str, str]
    agent_signals: dict[str, str]
    evidence_sources: list[str]
    conflicts: list[str]
    vetoes: list[str]
    pit_safe: bool
    data_quality_status: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision_time": self.decision_time.isoformat(),
            "final_action": self.final_action,
            "final_score": self.final_score,
            "final_confidence": self.final_confidence,
            "agent_statuses": dict(self.agent_statuses),
            "agent_signals": dict(self.agent_signals),
            "evidence_sources": list(self.evidence_sources),
            "conflicts": list(self.conflicts),
            "vetoes": list(self.vetoes),
            "pit_safe": self.pit_safe,
            "data_quality_status": self.data_quality_status,
            "rationale": list(self.rationale),
        }


class DecisionAuditService:
    @staticmethod
    def build(fusion_result, conviction_result, risk_result, final_decision, agent_results=None) -> DecisionAuditRecord:
        agents = agent_results or []
        statuses = {r.agent_name: r.status.value for r in agents}
        signals = {r.agent_name: r.signal.value for r in agents}
        evidence_sources = sorted({e.source for r in agents for e in getattr(r, "evidence", [])})
        vetoes = list(fusion_result.vetoes) + list(getattr(risk_result, "vetoes", []))
        return DecisionAuditRecord(
            symbol=fusion_result.symbol,
            decision_time=fusion_result.decision_time,
            final_action=final_decision["action"] if isinstance(final_decision, dict) else final_decision.action,
            final_score=float(final_decision["score"] if isinstance(final_decision, dict) else final_decision.score),
            final_confidence=float(final_decision["confidence"] if isinstance(final_decision, dict) else final_decision.confidence),
            agent_statuses=statuses,
            agent_signals=signals,
            evidence_sources=evidence_sources,
            conflicts=list(fusion_result.conflicts),
            vetoes=vetoes,
            pit_safe=bool(fusion_result.data_quality.pit_safe and risk_result.pit_safe),
            data_quality_status=fusion_result.data_quality.overall_status.value,
            rationale=list(final_decision["reasons"] if isinstance(final_decision, dict) else final_decision.reasons),
        )
