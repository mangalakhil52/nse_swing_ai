"""Tests for #14G Evidence Fusion orchestration."""
from datetime import date
import pytest
from src.architecture.evidence_fusion import EvidenceFusionService
from src.architecture.contracts import AgentAnalysisResult, StructuredEvidence
from src.core.models import SymbolMetadata
from src.core.types import AgentStatus, DataQualityStatus, SignalType


def _dq(pit_safe=True):
    from src.data.data_quality import DataQualityResult
    return DataQualityResult(overall_status=DataQualityStatus.OK, pit_safe=pit_safe, source_results=[])


def _result(symbol, agent, signal, pit=True):
    return AgentAnalysisResult(
        symbol=symbol, agent_name=agent, decision_time=date(2026, 6, 30),
        signal=signal, score=80, confidence=.8, pit_safe=pit,
        status=AgentStatus.SUCCESS,
        evidence=[StructuredEvidence(source=agent.upper(), observation=f"{agent}:{signal.value}",
                                      as_of=date(2026, 6, 30), direction=signal,
                                      strength="HIGH", reliability=.9, pit_safe=pit)],
    )


def test_fusion_preserves_directional_evidence_without_averaging():
    out = EvidenceFusionService.fuse(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), date(2026, 6, 30),
        [_result("TRENT", "technical", SignalType.BULLISH), _result("TRENT", "fundamental", SignalType.NEUTRAL)], _dq())
    assert len(out.bullish_evidence) == 1
    assert len(out.neutral_evidence) == 1
    assert out.aggregate_strength is None
    assert out.aggregate_confidence is None


def test_fusion_preserves_bull_bear_conflict():
    out = EvidenceFusionService.fuse(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), date(2026, 6, 30),
        [_result("TRENT", "technical", SignalType.BULLISH), _result("TRENT", "fundamental", SignalType.BEARISH)], _dq())
    assert out.conflicts
    assert any("CONTRADICTORY" in x for x in out.conflicts)


def test_pit_unsafe_agent_creates_veto():
    out = EvidenceFusionService.fuse(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), date(2026, 6, 30),
        [_result("TRENT", "technical", SignalType.BULLISH, pit=False)], _dq())
    assert "PIT_VIOLATION" in out.vetoes


def test_data_quality_pit_failure_creates_hard_veto():
    out = EvidenceFusionService.fuse(
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), date(2026, 6, 30),
        [_result("TRENT", "technical", SignalType.BULLISH)], _dq(pit_safe=False))
    assert "PIT_VIOLATION" in out.vetoes


def test_fusion_rejects_mixed_decision_times():
    r = _result("TRENT", "technical", SignalType.BULLISH)
    r.decision_time = date(2026, 6, 29)
    with pytest.raises(ValueError, match="decision_time"):
        EvidenceFusionService.fuse(SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), date(2026, 6, 30), [r], _dq())


def test_fusion_rejects_mixed_symbols():
    with pytest.raises(ValueError, match="symbol"):
        EvidenceFusionService.fuse(SymbolMetadata(symbol="TRENT", company_name="Trent Ltd"), date(2026, 6, 30), [_result("INFY", "technical", SignalType.BULLISH)], _dq())
