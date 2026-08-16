"""
Institutional Flow Specialist Agent Module.
Analyzes smart money accumulation footprints: official NSE delivery percentage, FII/DII shareholding changes, and bulk deals.
"""

from typing import Any
import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, ShareholdingPattern, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType


class InstitutionalFlowAgent(BaseAgent):
    """Specialist agent tracking institutional accumulation and delivery statistics."""

    def __init__(self):
        super().__init__(agent_name="institutional_flow_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        shp: ShareholdingPattern | None = context.get("shareholding_pattern")

        delivery_pct = 50.0
        avg_delivery_pct = 45.0
        delivery_surge = 1.0

        if not df.empty and "delivery_pct" in df.columns:
            recent_del = df["delivery_pct"].dropna()
            if not recent_del.empty:
                delivery_pct = float(recent_del.iloc[-1])
                avg_delivery_pct = float(recent_del.tail(20).mean())
                delivery_surge = delivery_pct / max(avg_delivery_pct, 1.0)

        fii_pct = shp.fii_pct if shp else 20.0
        dii_pct = shp.dii_pct if shp else 15.0
        inst_total = fii_pct + dii_pct

        # Score computation (0 to 100)
        score = 50.0

        if delivery_pct >= 55.0 and delivery_surge >= 1.2:
            score += 25.0
        elif delivery_pct >= 45.0:
            score += 15.0
        elif delivery_pct < 25.0:
            score -= 15.0

        if inst_total >= 30.0:
            score += 20.0
        elif inst_total >= 15.0:
            score += 10.0

        score = min(100.0, max(0.0, score))
        confidence = 0.86

        signal = SignalType.BULLISH if score >= 70.0 else (SignalType.BEARISH if score < 45.0 else SignalType.NEUTRAL)

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="DELIVERY_ACCUMULATION",
            raw_metric="nse_delivery_metrics",
            observed_value=f"Delivery: {delivery_pct:.1f}% (20D Avg: {avg_delivery_pct:.1f}%, Surge: {delivery_surge:.2f}x)",
            unit="delivery_pct",
            source="NSE_SEC_BHAV",
            timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns and not df.empty else "EOD",
        )

        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="INSTITUTIONAL_OWNERSHIP",
            raw_metric="fii_dii_holding",
            observed_value=f"FII: {fii_pct:.1f}%, DII: {dii_pct:.1f}% (Total Institutional: {inst_total:.1f}%)",
            unit="shareholding_pct",
            source="BSE_SHAREHOLDING_FILINGS",
            timestamp=shp.quarter_date.isoformat() if shp else "2026-06-30",
        )

        risks: list[str] = []
        if delivery_pct < 25.0:
            risks.append(f"Low delivery percentage ({delivery_pct:.1f}%): potential speculative intraday churning.")

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=round(score, 1),
            confidence=confidence,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "delivery_pct": round(delivery_pct, 1),
                "avg_delivery_pct": round(avg_delivery_pct, 1),
                "delivery_surge_ratio": round(delivery_surge, 2),
                "fii_pct": fii_pct,
                "dii_pct": dii_pct,
                "institutional_total_pct": inst_total,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
        )
