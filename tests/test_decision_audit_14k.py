"""#14K decision audit tests."""
from datetime import date
from types import SimpleNamespace
from src.architecture.decision_audit import DecisionAuditService


def _fusion():
    return SimpleNamespace(
        symbol="TRENT", decision_time=date(2026, 6, 30),
        vetoes=[], conflicts=["technical_vs_fundamental"],
        data_quality=SimpleNamespace(pit_safe=True, overall_status=SimpleNamespace(value="VALID")),
    )


def _risk():
    return SimpleNamespace(pit_safe=True, vetoes=[])


def _decision():
    return {"action": "BUY", "score": 85.0, "confidence": 0.85,
            "reasons": ["HIGH_CONVICTION", "RISK_OK"]}


def test_audit_is_reproducible_and_serializable():
    agents = [
        SimpleNamespace(agent_name="technical", status=SimpleNamespace(value="SUCCESS"),
                         signal=SimpleNamespace(value="BULLISH"),
                         evidence=[SimpleNamespace(source="TECHNICAL")]),
        SimpleNamespace(agent_name="fundamental", status=SimpleNamespace(value="SUCCESS"),
                         signal=SimpleNamespace(value="BULLISH"),
                         evidence=[SimpleNamespace(source="FUNDAMENTAL")]),
    ]
    a = DecisionAuditService.build(_fusion(), SimpleNamespace(), _risk(), _decision(), agents)
    b = DecisionAuditService.build(_fusion(), SimpleNamespace(), _risk(), _decision(), agents)
    assert a == b
    assert a.to_dict()["decision_time"] == "2026-06-30"
    assert a.to_dict()["final_action"] == "BUY"


def test_audit_preserves_agent_and_conflict_information():
    agents = [SimpleNamespace(agent_name="news", status=SimpleNamespace(value="SUCCESS"),
                              signal=SimpleNamespace(value="NEUTRAL"),
                              evidence=[SimpleNamespace(source="NEWS")])]
    a = DecisionAuditService.build(_fusion(), SimpleNamespace(), _risk(), _decision(), agents)
    assert a.agent_statuses == {"news": "SUCCESS"}
    assert a.agent_signals == {"news": "NEUTRAL"}
    assert a.evidence_sources == ["NEWS"]
    assert a.conflicts == ["technical_vs_fundamental"]


def test_audit_marks_pit_failure():
    fusion = _fusion()
    fusion.data_quality.pit_safe = False
    agents = []
    a = DecisionAuditService.build(fusion, SimpleNamespace(), _risk(), _decision(), agents)
    assert a.pit_safe is False
