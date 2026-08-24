"""#14H conviction synthesis tests."""
from datetime import date
from src.architecture.conviction_engine import ConvictionSynthesisService
from src.architecture.contracts import ConvictionGrade, FusionResult, StructuredEvidence
from src.core.types import SignalType
from src.data.data_quality import DataQualityResult, DataQualityStatus


def _dq(pit=True):
    return DataQualityResult(symbol="TRENT", as_of_date=date(2026,6,30), overall_status=DataQualityStatus.VALID if pit else DataQualityStatus.PIT_VIOLATION,
        overall_quality_score=100.0 if pit else 0.0, pit_safe=pit, is_trade_eligible=pit)

def _e(source, direction):
    return StructuredEvidence(source=source, observation=f"{source}:{direction.value}", as_of=date(2026,6,30), direction=direction, strength="HIGH", reliability=.9, pit_safe=True)

def _fusion(b=0, bear=0, neutral=0, unknown=0, vetoes=None, pit=True):
    return FusionResult(symbol="TRENT", decision_time=date(2026,6,30),
        bullish_evidence=[_e("T",SignalType.BULLISH) for _ in range(b)],
        bearish_evidence=[_e("F",SignalType.BEARISH) for _ in range(bear)],
        neutral_evidence=[_e("N",SignalType.NEUTRAL) for _ in range(neutral)],
        unknown_evidence=[_e("U",SignalType.UNKNOWN) for _ in range(unknown)],
        conflicts=[], aggregate_strength=None, aggregate_confidence=None, data_quality=_dq(pit), vetoes=vetoes or [])

def test_veto_always_rejects():
    out=ConvictionSynthesisService.evaluate(_fusion(b=4,vetoes=["PIT_VIOLATION"]))
    assert out.grade == ConvictionGrade.REJECTED and out.score == 0

def test_pit_failure_always_rejects():
    out=ConvictionSynthesisService.evaluate(_fusion(b=4,pit=False))
    assert out.grade == ConvictionGrade.REJECTED

def test_no_directional_evidence_not_computed():
    out=ConvictionSynthesisService.evaluate(_fusion(neutral=2,unknown=1))
    assert out.grade == ConvictionGrade.NOT_COMPUTED

def test_conflict_is_low_conviction():
    out=ConvictionSynthesisService.evaluate(_fusion(b=2,bear=1))
    assert out.grade == ConvictionGrade.LOW_CONVICTION
    assert out.score == 30

def test_strong_directional_evidence_is_high_conviction():
    out=ConvictionSynthesisService.evaluate(_fusion(b=4))
    assert out.grade == ConvictionGrade.HIGH_CONVICTION
    assert out.score == 100

def test_unknown_caps_conviction():
    out=ConvictionSynthesisService.evaluate(_fusion(b=4,unknown=1))
    assert out.score <= 69.9
    assert out.grade == ConvictionGrade.MEDIUM_CONVICTION
