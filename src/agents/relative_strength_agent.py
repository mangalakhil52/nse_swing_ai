"""
Relative Strength Specialist Agent Module.
Calculates Mansfield Relative Strength, benchmark alpha against NIFTY 50 and Sector Index, and universe percentile score.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType
from src.quant.relative_strength import RelativeStrengthEngine


class RelativeStrengthAgent(BaseAgent):
    """Specialist agent analyzing relative leadership and alpha generation."""

    def __init__(self):
        super().__init__(agent_name="relative_strength_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        nifty_df: pd.DataFrame | None = context.get("nifty_df")
        sector_df: pd.DataFrame | None = context.get("sector_df")
        universe_rs_scores: dict[str, float] = context.get("universe_rs_scores", {})

        if df.empty or nifty_df is None or nifty_df.empty:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.NEUTRAL,
                score=50.0,
                confidence=0.5,
            )

        # Calculate Mansfield RS vs Nifty 50
        mansfield_series = RelativeStrengthEngine.calculate_mansfield_rs(df["close"], nifty_df["close"], period=50)
        current_rs = float(mansfield_series.iloc[-1])

        # Calculate multi-period alpha
        alphas = RelativeStrengthEngine.calculate_multi_period_alpha(df["close"], nifty_df["close"])

        # Calculate percentile rank
        percentiles = RelativeStrengthEngine.calculate_universe_percentile_ranks(universe_rs_scores)
        percentile_rank = percentiles.get(symbol, 75.0)

        # Score computation (0 to 100)
        score = 50.0
        if current_rs > 5.0:
            score += 25.0
        elif current_rs > 0.0:
            score += 15.0
        else:
            score -= 20.0

        if percentile_rank >= 85.0:
            score += 15.0
        elif percentile_rank >= 70.0:
            score += 8.0

        if alphas.get("alpha_20d", 0.0) > 5.0:
            score += 10.0

        score = min(100.0, max(0.0, score))
        confidence = 0.94

        signal = SignalType.BULLISH if score >= 75.0 else (SignalType.BEARISH if score < 45.0 else SignalType.NEUTRAL)

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="RELATIVE_STRENGTH",
            raw_metric="mansfield_rs_50",
            observed_value=round(current_rs, 2),
            unit="pct_outperformance",
            source="QUANT_ENGINE",
            timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns else "EOD",
        )

        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="RELATIVE_STRENGTH",
            raw_metric="universe_rs_percentile",
            observed_value=f"{percentile_rank:.1f}th Percentile",
            unit="percentile",
            source="QUANT_ENGINE",
            timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns else "EOD",
        )

        risks: list[str] = []
        if current_rs < 0.0:
            risks.append(f"Stock is lagging NIFTY 50 (Mansfield RS: {current_rs:.2f}%)")

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
                "mansfield_rs": round(current_rs, 2),
                "percentile_rank": round(percentile_rank, 1),
                "alpha_5d": alphas.get("alpha_5d", 0.0),
                "alpha_20d": alphas.get("alpha_20d", 0.0),
                "alpha_60d": alphas.get("alpha_60d", 0.0),
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
        )
