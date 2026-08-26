"""Specialist-agent orchestration with explicit evidence and fail-closed semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.agents.contracts import Agent, AgentResult
from src.runtime.telemetry import agent, alert


@dataclass(frozen=True)
class OrchestrationResult:
    symbol: str
    results: tuple[AgentResult, ...]
    final_decision: str


class AgentOrchestrator:
    """Run registered agents without inventing missing intelligence."""

    def __init__(self, agents: Sequence[Agent] = ()) -> None:
        self.agents = tuple(agents)

    def evaluate(self, symbol: str, context: Mapping[str, Any]) -> OrchestrationResult:
        results: list[AgentResult] = []
        for specialist in self.agents:
            agent(specialist.name, status="EVALUATING", progress=0, processed=0, decision=f"Evaluating {symbol}")
            try:
                result = specialist.evaluate(symbol, context)
            except Exception as exc:
                alert(f"{specialist.name} failed for {symbol}: {type(exc).__name__}", "red")
                result = AgentResult(
                    agent=specialist.name,
                    symbol=symbol,
                    status="ERROR",
                    decision="REJECT",
                    reason="Agent execution failed; fail-closed",
                )
            results.append(result)
            agent(specialist.name, status=result.status, progress=100, processed=1, decision=result.decision, log=[result.reason] if result.reason else [])

        # No agent is allowed to imply BUY/SELL unless an explicit downstream
        # CIO/Risk agent has produced that decision. Missing specialists remain WAIT.
        final = next((r.decision for r in reversed(results) if r.agent == "CIO"), "WAIT")
        return OrchestrationResult(symbol=symbol, results=tuple(results), final_decision=final)
