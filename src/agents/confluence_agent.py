"""
Confluence and Disagreement Detection Specialist Agent Module.
Aggregates independent agent outputs, evaluates structural alignment, and detects logical contradictions across desks.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import (
    AgentStatus,
    ConfluenceState,
    DataFreshness,
    SignalType,
)


class ConfluenceAgent(BaseAgent):
    """Evaluates cross-agent alignment and detects logical disagreements."""

    def __init__(self):
        super().__init__(agent_name="confluence_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        agent_outputs: dict[str, AgentOutput] = context.get("agent_outputs", {})

        tech = agent_outputs.get("technical_analysis_agent")
        rs = agent_outputs.get("relative_strength_agent")
        fund = agent_outputs.get("fundamental_analysis_agent")
        news = agent_outputs.get("news_intelligence_agent")
        forensic = agent_outputs.get("forensic_analysis_agent")
        risk = agent_outputs.get("risk_management_agent")

        # 1. Disagreement & Contradiction Detection
        conflicted = False
        disagreements: list[str] = []

        if tech and tech.signal == SignalType.BULLISH and news and news.signal == SignalType.BEARISH:
            conflicted = True
            disagreements.append("Technical Bullish vs News Bearish Contradiction")

        if tech and tech.signal == SignalType.BULLISH and forensic and forensic.signal == SignalType.REJECT:
            conflicted = True
            disagreements.append("Technical Bullish vs Forensic Disqualification Contradiction")

        if risk and risk.disqualification_triggered:
            conflicted = True
            disagreements.append(f"Risk Management Disqualification: {risk.disqualification_reason}")

        # 2. Confluence State Classification
        bull_signals = sum(1 for a in agent_outputs.values() if a.signal == SignalType.BULLISH)
        total_valid = sum(1 for a in agent_outputs.values() if a.status == AgentStatus.SUCCESS)

        if conflicted:
            state = ConfluenceState.CONFLICTED
            score = 30.0
            signal = SignalType.REJECT
        elif bull_signals >= 5:
            state = ConfluenceState.VERY_HIGH
            score = 95.0
            signal = SignalType.BULLISH
        elif bull_signals >= 4:
            state = ConfluenceState.HIGH
            score = 85.0
            signal = SignalType.BULLISH
        elif bull_signals >= 3:
            state = ConfluenceState.MODERATE
            score = 70.0
            signal = SignalType.NEUTRAL
        else:
            state = ConfluenceState.LOW
            score = 45.0
            signal = SignalType.NEUTRAL

        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="CONFLUENCE_SYNTHESIS",
            raw_metric="cross_agent_alignment",
            observed_value=f"State: {state.value} ({bull_signals}/{total_valid} Bullish Signals)",
            unit="confluence",
            source="CONFLUENCE_ENGINE",
            timestamp="EOD",
        )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=score,
            confidence=0.95,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "confluence_state": state.value,
                "bullish_signals_count": bull_signals,
                "total_valid_agents": total_valid,
                "disagreements": disagreements,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=disagreements,
            disqualification_triggered=conflicted,
            disqualification_reason="; ".join(disagreements) if conflicted else None,
        )
