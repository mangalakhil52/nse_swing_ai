from src.agents.contracts import Agent, AgentResult
from src.agents.orchestrator import AgentOrchestrator


class StubAgent(Agent):
    name = "NEWS"

    def evaluate(self, symbol, context):
        return AgentResult(agent=self.name, symbol=symbol, status="COMPLETE", score=80, confidence=0.9, decision="PASS", reason="No blocking event")


def test_orchestrator_returns_wait_without_cio():
    result = AgentOrchestrator([StubAgent()]).evaluate("TRENT", {})
    assert result.final_decision == "WAIT"
    assert result.results[0].decision == "PASS"


def test_orchestrator_fail_closes_agent_errors():
    class Broken(Agent):
        name = "FUNDAMENTAL"
        def evaluate(self, symbol, context):
            raise RuntimeError("boom")
    result = AgentOrchestrator([Broken()]).evaluate("TRENT", {})
    assert result.final_decision == "WAIT"
    assert result.results[0].status == "ERROR"
    assert result.results[0].decision == "REJECT"
