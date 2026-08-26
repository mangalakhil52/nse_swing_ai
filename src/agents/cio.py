"""CIO fusion: final decision layer, never bypassing risk."""
from __future__ import annotations

from typing import Sequence
from src.agents.contracts import Agent, AgentEvidence, AgentResult


class CIOAgent(Agent):
    name = "CIO"

    def evaluate(self, symbol: str, context: dict) -> AgentResult:
        results: Sequence[AgentResult] = tuple(context.get("specialist_results", ()))
        risk = next((r for r in results if r.agent == "RISK"), None)
        non_risk = [r for r in results if r.agent not in {"RISK", "CIO"}]
        if risk is None or risk.decision != "PASS":
            return AgentResult(self.name, symbol, "COMPLETE", decision="WAIT", reason="CIO requires an explicit PASS from Risk")
        valid = [r for r in non_risk if r.status == "COMPLETE" and r.confidence is not None]
        if not valid:
            return AgentResult(self.name, symbol, "WAIT", decision="WAIT", reason="No specialist evidence")
        confidence = sum(float(r.confidence) for r in valid) / len(valid)
        positives = sum(r.decision == "PASS" for r in valid)
        decision = "BUY" if positives >= 2 and confidence >= 0.70 else "WATCH"
        return AgentResult(
            self.name, symbol, "COMPLETE", score=confidence * 100, confidence=confidence,
            decision=decision,
            evidence=tuple(AgentEvidence("specialist_decision", r.decision, r.agent, r.generated_at, "POSITIVE" if r.decision == "PASS" else "NEUTRAL") for r in valid),
            reason=f"{positives}/{len(valid)} specialist agents passed; risk passed",
        )
