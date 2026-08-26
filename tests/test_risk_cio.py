from src.agents.contracts import AgentResult
from src.agents.risk import RiskAgent
from src.agents.cio import CIOAgent


def _specialist(name, decision="PASS", confidence=.8):
    return AgentResult(name, "TRENT", "COMPLETE", confidence=confidence, decision=decision)


def test_risk_fails_closed_without_confidence():
    out = RiskAgent().evaluate("TRENT", {"specialist_results": []})
    assert out.decision == "WAIT"


def test_risk_rejects_low_confidence():
    out = RiskAgent().evaluate("TRENT", {"specialist_results": [_specialist("NEWS", confidence=.4)]})
    assert out.decision == "REJECT"


def test_cio_requires_risk_pass():
    out = CIOAgent().evaluate("TRENT", {"specialist_results": [_specialist("TECHNICAL"), _specialist("NEWS")]})
    assert out.decision == "WAIT"


def test_cio_can_buy_only_after_risk_and_two_specialists_pass():
    risk = RiskAgent().evaluate("TRENT", {"specialist_results": [_specialist("NEWS", confidence=.8), _specialist("TECHNICAL", confidence=.9)]})
    out = CIOAgent().evaluate("TRENT", {"specialist_results": [_specialist("NEWS"), _specialist("TECHNICAL"), risk]})
    assert out.decision == "BUY"
