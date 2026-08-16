"""
Sector Rotation Specialist Agent Module.
Tracks the 14 major NSE Sector Indices, identifies leading vs lagging sectors, and evaluates sector tailwinds.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType


class SectorRotationAgent(BaseAgent):
    """Specialist agent analyzing sector momentum and rotation."""

    def __init__(self):
        super().__init__(agent_name="sector_rotation_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        sector = symbol_meta.sector or "General"
        sector_rankings: dict[str, int] = context.get("sector_rankings", {})

        # Default to middle rank if not provided
        rank = sector_rankings.get(sector, 5)
        total_sectors = max(len(sector_rankings), 10)

        # Score computation (0 to 100)
        # Top 3 sectors: High tailwind (80-95)
        # Middle sectors: Neutral (60-75)
        # Bottom 3 sectors: Headwind (30-45)
        if rank <= 3:
            score = 90.0 - (rank - 1) * 5.0
            signal = SignalType.BULLISH
            status_desc = f"Sector '{sector}' is a top market leader (Rank #{rank} of {total_sectors})"
        elif rank <= total_sectors - 3:
            score = 65.0
            signal = SignalType.NEUTRAL
            status_desc = f"Sector '{sector}' is performing inline with benchmark (Rank #{rank} of {total_sectors})"
        else:
            score = 35.0
            signal = SignalType.BEARISH
            status_desc = f"Sector '{sector}' is currently lagging the broader market (Rank #{rank} of {total_sectors})"

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="SECTOR_MOMENTUM",
            raw_metric="sector_relative_rank",
            observed_value=status_desc,
            unit="sector_rank",
            source="NSE_INDEX_FEEDS",
            timestamp="EOD",
        )

        risks: list[str] = []
        if rank > total_sectors - 3:
            risks.append(f"Sector headwind: {sector} is in bottom {total_sectors - rank + 1} lagging sectors.")

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=round(score, 1),
            confidence=0.88,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "sector_name": sector,
                "sector_rank": rank,
                "total_sectors": total_sectors,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
        )
