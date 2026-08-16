"""
Base Agent Module for Specialist Research Desks.
Provides execution lifecycle telemetry, structured JSON contract enforcement, evidence registration, and exception resilience.
"""

from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
import logging
import time
from typing import Any
import pandas as pd

from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, EvidenceItem, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all domain-specific research agents."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def execute(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """
        Wrapper handling timing, error catching, and structured output creation.
        """
        start_time = time.perf_counter()
        symbol = symbol_meta.symbol.upper().strip()
        context = context or {}

        try:
            output = await self._analyze(symbol_meta, df, evidence_graph, run_id, context)
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            output.execution_time_ms = exec_time_ms
            output.run_id = run_id
            output.timestamp = datetime.utcnow()
            return output
        except Exception as e:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Agent '{self.agent_name}' failed for {symbol}: {e}", exc_info=True)
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                timestamp=datetime.utcnow(),
                status=AgentStatus.FAILED,
                signal=SignalType.NEUTRAL,
                score=0.0,
                confidence=0.0,
                data_freshness=DataFreshness.UNKNOWN,
                risks_identified=[f"Agent execution failure: {str(e)}"],
                execution_time_ms=exec_time_ms,
            )

    @abstractmethod
    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        """Domain-specific analysis logic implemented by subclasses."""
        pass
