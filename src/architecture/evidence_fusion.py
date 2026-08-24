"""#14G Evidence Fusion orchestration layer.

Deliberately thin: specialist agents own analysis; this layer only validates,
classifies, and preserves evidence before conviction/risk/CIO synthesis.
"""
from datetime import date, datetime
from src.architecture.contracts import AgentAnalysisResult, EvidenceFusionEngine, FusionResult
from src.core.models import SymbolMetadata


class EvidenceFusionService:
    """Production entry point for deterministic multi-agent evidence fusion."""

    @staticmethod
    def fuse(
        symbol_meta: SymbolMetadata,
        decision_time: datetime | date,
        agent_results: list[AgentAnalysisResult],
        data_quality,
    ) -> FusionResult:
        if any(r.decision_time != decision_time for r in agent_results):
            raise ValueError("All agent results must use the same decision_time")
        if any(r.symbol != symbol_meta.symbol for r in agent_results):
            raise ValueError("All agent results must belong to the requested symbol")
        return EvidenceFusionEngine.fuse_evidence(
            symbol=symbol_meta.symbol,
            decision_time=decision_time,
            agent_results=agent_results,
            data_quality=data_quality,
        )
