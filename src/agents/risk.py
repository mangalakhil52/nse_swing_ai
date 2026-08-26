"""Risk gate: converts specialist outputs into explicit, conservative constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.agents.contracts import Agent, AgentEvidence, AgentResult


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 0.05
    max_risk_per_trade_pct: float = 1.0
    min_confidence: float = 0.60
    max_stop_distance_pct: float = 8.0


class RiskAgent(Agent):
    name = "RISK"

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(self, symbol: str, context: dict) -> AgentResult:
        specialists: Sequence[AgentResult] = tuple(context.get("specialist_results", ()))
        confidence_values = [r.confidence for r in specialists if r.confidence is not None]
        confidence = min(confidence_values) if confidence_values else None
        stop_distance = context.get("stop_distance_pct")
        risks: list[str] = []
        if confidence is None:
            return AgentResult(self.name, symbol, "WAIT", decision="WAIT", reason="No specialist confidence available")
        if confidence < self.limits.min_confidence:
            risks.append("CONFIDENCE_BELOW_RISK_THRESHOLD")
        if stop_distance is not None and float(stop_distance) > self.limits.max_stop_distance_pct:
            risks.append("STOP_DISTANCE_TOO_WIDE")
        decision = "REJECT" if risks else "PASS"
        return AgentResult(
            agent=self.name, symbol=symbol, status="COMPLETE", score=confidence * 100,
            confidence=confidence, decision=decision,
            evidence=(AgentEvidence("min_specialist_confidence", confidence, "agent_outputs", str(context.get("as_of", "")), "POSITIVE" if not risks else "NEGATIVE"),),
            risks=tuple(risks), reason="Risk limits passed" if not risks else "; ".join(risks),
        )
