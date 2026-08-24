"""#14J CIO/final decision synthesis."""
from src.architecture.contracts import ConvictionGrade, FusionResult, RiskEngineResult


class CIODecisionService:
    @staticmethod
    def decide(fusion_result: FusionResult, conviction_result, risk_result: RiskEngineResult):
        vetoes = list(fusion_result.vetoes) + list(getattr(risk_result, "vetoes", []))
        if vetoes or not fusion_result.data_quality.pit_safe or not risk_result.pit_safe:
            return {"symbol": fusion_result.symbol, "decision_time": fusion_result.decision_time,
                    "action": "REJECTED", "score": 0.0, "confidence": 0.0,
                    "reasons": ["HARD_VETO_PRESENT"] + vetoes}
        grade = conviction_result.grade
        if grade == ConvictionGrade.HIGH_CONVICTION and getattr(risk_result, "passed_risk_veto", False):
            action = "BUY"
        elif grade in {ConvictionGrade.MEDIUM_CONVICTION, ConvictionGrade.LOW_CONVICTION}:
            action = "WATCH"
        else:
            action = "AVOID"
        score = float(getattr(conviction_result, "score", 0.0))
        reasons = list(getattr(conviction_result, "reasons", []))
        reasons.extend(getattr(risk_result, "reasons", []))
        return {"symbol": fusion_result.symbol, "decision_time": fusion_result.decision_time,
                "action": action, "score": score, "confidence": min(score / 100.0, 1.0),
                "reasons": reasons}
