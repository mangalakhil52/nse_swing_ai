"""#14J CIO/final decision synthesis.

Consumes only validated upstream outputs. Hard vetoes are terminal; the CIO
cannot manufacture missing evidence, override PIT failures, or perform sizing.
"""
from src.architecture.contracts import ConvictionGrade, FinalDecision, FusionResult, RiskEngineResult


class CIODecisionService:
    @staticmethod
    def decide(fusion_result: FusionResult, conviction_result, risk_result: RiskEngineResult) -> FinalDecision:
        vetoes = list(fusion_result.vetoes) + list(getattr(risk_result, "vetoes", []))
        if vetoes or not fusion_result.data_quality.pit_safe or not risk_result.pit_safe:
            return FinalDecision(
                symbol=fusion_result.symbol,
                decision_time=fusion_result.decision_time,
                action="REJECTED",
                score=0.0,
                confidence=0.0,
                reasons=["HARD_VETO_PRESENT"] + vetoes,
            )

        grade = conviction_result.grade
        if grade == ConvictionGrade.HIGH_CONVICTION and getattr(risk_result, "passed_risk_veto", False):
            action = "BUY"
        elif grade in {ConvictionGrade.MEDIUM_CONVICTION, ConvictionGrade.LOW_CONVICTION}:
            action = "WATCH"
        else:
            action = "AVOID"

        confidence = min(float(getattr(conviction_result, "score", 0.0)) / 100.0,
                         float(getattr(conviction_result, "confidence", 1.0)))
        reasons = list(getattr(conviction_result, "reasons", []))
        reasons.extend(getattr(risk_result, "reasons", []))
        return FinalDecision(
            symbol=fusion_result.symbol,
            decision_time=fusion_result.decision_time,
            action=action,
            score=float(getattr(conviction_result, "score", 0.0)),
            confidence=confidence,
            reasons=reasons,
        )
