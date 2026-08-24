"""#14H deterministic conviction synthesis."""
from src.architecture.contracts import ConvictionGrade, ConvictionResult, FusionResult


class ConvictionSynthesisService:
    """Conservative, explainable conviction synthesis from FusionResult."""

    @staticmethod
    def evaluate(fusion_result: FusionResult) -> ConvictionResult:
        if fusion_result.vetoes or not fusion_result.data_quality.pit_safe:
            return ConvictionResult(symbol=fusion_result.symbol, decision_time=fusion_result.decision_time,
                grade=ConvictionGrade.REJECTED, score=0.0,
                reasons=["REJECTED_DUE_TO_VETOES_OR_PIT_FAILURE"])

        bullish = len(fusion_result.bullish_evidence)
        bearish = len(fusion_result.bearish_evidence)
        neutral = len(fusion_result.neutral_evidence)
        unknown = len(fusion_result.unknown_evidence)
        total = bullish + bearish + neutral + unknown

        if total == 0 or (unknown > 0 and bullish == 0 and bearish == 0):
            return ConvictionResult(symbol=fusion_result.symbol, decision_time=fusion_result.decision_time,
                grade=ConvictionGrade.NOT_COMPUTED, score=0.0,
                reasons=["INSUFFICIENT_DIRECTIONAL_EVIDENCE"])

        if bullish and bearish:
            return ConvictionResult(symbol=fusion_result.symbol, decision_time=fusion_result.decision_time,
                grade=ConvictionGrade.LOW_CONVICTION, score=30.0,
                reasons=["CONTRADICTORY_SIGNALS_REQUIRE_LOW_CONVICTION"])

        directional = bullish if bullish else bearish
        direction_share = directional / max(total, 1)
        score = round(100.0 * direction_share, 1)
        if unknown:
            score = min(score, 69.9)

        if score >= 80.0:
            grade = ConvictionGrade.HIGH_CONVICTION
        elif score >= 60.0:
            grade = ConvictionGrade.MEDIUM_CONVICTION
        else:
            grade = ConvictionGrade.LOW_CONVICTION

        direction = "BULLISH" if bullish else "BEARISH"
        return ConvictionResult(symbol=fusion_result.symbol, decision_time=fusion_result.decision_time,
            grade=grade, score=score,
            reasons=[f"{direction}_EVIDENCE_SHARE={direction_share:.2f}",
                     f"directional={directional}, neutral={neutral}, unknown={unknown}"])
