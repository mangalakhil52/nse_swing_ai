"""
Thesis Killer / Devil's Advocate Specialist Agent Module.
Actively seeks counter-evidence, structural vulnerabilities, and hidden flaws in candidate trade setups.
Enforces a hard Thesis Fragility Score threshold to shoot down flawed trade ideas before capital is deployed.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType


class ThesisKillerAgent(BaseAgent):
    """Devil's Advocate agent assessing thesis fragility, hidden risks, and counter-evidence."""

    def __init__(self):
        super().__init__(agent_name="thesis_killer_agent")

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

        # Extract sub-agent metrics to evaluate fragility
        tech_out = agent_outputs.get("technical_analysis_agent")
        fund_out = agent_outputs.get("fundamental_analysis_agent")
        news_out = agent_outputs.get("news_intelligence_agent")
        risk_out = agent_outputs.get("risk_management_agent")

        fragility_score = 0.0
        flaws_identified: list[str] = []

        # 1. Check for Volume Distribution / Selling Climax
        if not df.empty and len(df) >= 20:
            latest_bar = df.iloc[-1]
            avg_vol = df["volume"].tail(20).mean()
            # Red candle with 2.5x volume = heavy distribution
            if latest_bar["close"] < latest_bar["open"] and latest_bar["volume"] > 2.5 * avg_vol:
                fragility_score += 25.0
                flaws_identified.append("Heavy volume distribution detected on latest red candle (> 2.5x RVol).")

        # 2. Check for Margin Compression & Accrual Earnings
        if fund_out and fund_out.metrics:
            fcf_pat = fund_out.metrics.get("fcf_to_pat", 1.0)
            if fcf_pat < 0.50:
                fragility_score += 25.0
                flaws_identified.append(f"Accrual earnings risk: Free Cash Flow to PAT conversion is weak ({fcf_pat:.2f}).")

            pat_accel = fund_out.metrics.get("earnings_acceleration", 0.0)
            if pat_accel < -15.0:
                fragility_score += 15.0
                flaws_identified.append(f"Severe earnings deceleration: YoY PAT growth slowed by {abs(pat_accel):.1f}%.")

        # 3. Check for Negative News / Filing Warnings
        if news_out and news_out.metrics:
            neg_articles = news_out.metrics.get("negative_articles", 0)
            if neg_articles > 0:
                fragility_score += 20.0
                flaws_identified.append(f"Negative news presence: {neg_articles} negative headlines in 7-day window.")

        # 4. Check for Stop Loss Distance Fragility
        if risk_out and risk_out.metrics:
            sl_pct = risk_out.metrics.get("stop_loss_pct", 5.0)
            if sl_pct > 7.5:
                fragility_score += 15.0
                flaws_identified.append(f"Wide stop-loss distance ({sl_pct:.1f}%), increasing position fragility.")

        fragility_score = min(100.0, max(0.0, fragility_score))
        disqualified = fragility_score >= 55.0

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="THESIS_FRAGILITY",
            raw_metric="fragility_score",
            observed_value=f"Fragility Score: {fragility_score:.1f}/100 | Flaws: {len(flaws_identified)}",
            unit="fragility_index",
            source="DEVILS_ADVOCATE_ENGINE",
            timestamp=pd.Timestamp.now().isoformat(),
        )

        signal = SignalType.BEARISH if fragility_score >= 50.0 else SignalType.NEUTRAL

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=round(100.0 - fragility_score, 1),
            confidence=0.88,
            data_freshness=DataFreshness.RECENT,
            disqualification_triggered=disqualified,
            disqualification_reason=f"THESIS KILLED: Fragility score {fragility_score:.1f} >= 65.0 ({'; '.join(flaws_identified[:2])})" if disqualified else None,
            metrics={
                "fragility_score": fragility_score,
                "flaws_identified_count": len(flaws_identified),
                "thesis_killed": disqualified,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=flaws_identified,
        )
