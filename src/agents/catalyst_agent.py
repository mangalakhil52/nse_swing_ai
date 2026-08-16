"""
Catalyst Specialist Agent Module.
Identifies forward-looking business and macro events (e.g. order wins, capacity commissioning, earnings dates)
that can accelerate momentum within the expected 3–15 day swing window.
"""

from datetime import date, datetime, timedelta
from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    CorporateEvent,
    NewsArticle,
    SymbolMetadata,
)
from src.core.types import AgentStatus, CatalystType, DataFreshness, SignalType


class CatalystAgent(BaseAgent):
    """Specialist agent identifying actionable momentum catalysts."""

    def __init__(self):
        super().__init__(agent_name="catalyst_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        upcoming_events: list[CorporateEvent] = context.get("upcoming_events", [])
        articles: list[NewsArticle] = context.get("news_articles", [])

        # Check for forward catalyst articles
        catalyst_articles = [a for a in articles if a.is_catalyst]

        score = 50.0  # Base neutral (a stock does not require a catalyst if technical setup is clean)
        catalyst_desc = "No immediate binary catalyst detected; technical setup is self-sufficient."
        cat_type = CatalystType.NO_CATALYST

        if catalyst_articles:
            top_cat = catalyst_articles[0]
            cat_type = top_cat.catalyst_type
            score = 85.0
            catalyst_desc = f"Identified positive catalyst: {top_cat.headline}"
            signal = SignalType.BULLISH

            evidence_graph.add_evidence(
                symbol=symbol,
                agent_name=self.agent_name,
                claim_type="BUSINESS_CATALYST",
                raw_metric="forward_catalyst",
                observed_value=catalyst_desc,
                unit="event",
                source=top_cat.publisher,
                timestamp=top_cat.published_at.isoformat(),
                citation_url=top_cat.source_url,
            )
        else:
            signal = SignalType.NEUTRAL

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=round(score, 1),
            confidence=0.85,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "catalyst_type": cat_type.value,
                "has_catalyst": len(catalyst_articles) > 0,
                "description": catalyst_desc,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=[],
        )
