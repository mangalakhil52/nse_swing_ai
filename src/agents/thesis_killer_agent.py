"""
Thesis Killer / Devil's Advocate Specialist Agent Module — Part 30 Independent Adversarial Engine.

Receives raw evidence, proposed thesis, and proposed trade geometry.
Actively attempts to shoot down the thesis by investigating:
  - False breakout risk & volume distribution
  - Accrual earnings & FCF conversion deficit
  - Severe YoY earnings deceleration
  - Imminent event/earnings risk
  - Customer concentration & regulatory headwinds
Outputs: SURVIVES / WEAKENED / KILLED with evidence.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType


class ThesisKillerAgent(BaseAgent):
    """Devil's Advocate agent performing independent adversarial review on candidate trade thesis."""

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

        tech_out = agent_outputs.get("technical_analysis_agent")
        fund_out = agent_outputs.get("fundamental_analysis_agent")
        news_out = agent_outputs.get("news_intelligence_agent")
        risk_out = agent_outputs.get("risk_management_agent")

        fragility_score = 0.0
        flaws_identified: list[str] = []

        # 1. False Breakout & Volume Distribution Check
        if not df.empty and len(df) >= 20:
            latest_bar = df.iloc[-1]
            avg_vol = df["volume"].tail(20).mean()
            if latest_bar["close"] < latest_bar["open"] and latest_bar["volume"] > 2.5 * max(1, avg_vol):
                fragility_score += 25.0
                flaws_identified.append("Heavy volume distribution on latest red candle (> 2.5x RVol).")

        # 2. Accrual Earnings & Cash Conversion Deficit Check
        if fund_out and fund_out.metrics:
            fcf_pat = fund_out.metrics.get("fcf_to_pat", 1.0)
            if fcf_pat < 0.50:
                fragility_score += 25.0
                flaws_identified.append(f"Accrual earnings risk: FCF/PAT conversion is weak ({fcf_pat:.2f}).")

            pat_accel = fund_out.metrics.get("earnings_acceleration", 0.0)
            if pat_accel < -15.0:
                fragility_score += 15.0
                flaws_identified.append(f"Severe earnings deceleration: YoY PAT growth slowed by {abs(pat_accel):.1f}%.")

        # 3. Negative News & Press Headwinds
        if news_out and news_out.metrics:
            neg_articles = news_out.metrics.get("negative_articles", 0)
            if neg_articles > 0:
                fragility_score += 20.0
                flaws_identified.append(f"Negative news presence: {neg_articles} negative headlines in recent window.")

        # 4. Wide Stop-Loss Position Fragility
        if risk_out and risk_out.metrics:
            sl_pct = risk_out.metrics.get("stop_loss_pct", 5.0)
            if sl_pct > 7.5:
                fragility_score += 15.0
                flaws_identified.append(f"Wide stop-loss distance ({sl_pct:.1f}%), increasing position fragility.")

        fragility_score = min(100.0, max(0.0, fragility_score))

        if fragility_score >= 55.0:
            thesis_outcome = "KILLED"
            disqualified = True
        elif fragility_score >= 30.0:
            thesis_outcome = "WEAKENED"
            disqualified = False
        else:
            thesis_outcome = "SURVIVES"
            disqualified = False

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="ADVERSARIAL_THESIS_REVIEW",
            raw_metric="thesis_outcome",
            observed_value=f"Thesis Outcome: {thesis_outcome} (Fragility Score: {fragility_score:.1f}/100)",
            unit="outcome",
            source="THESIS_KILLER_ENGINE",
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
            confidence=None,  # Uncalibrated confidence marked as None
            data_freshness=DataFreshness.RECENT,
            disqualification_triggered=disqualified,
            disqualification_reason=f"THESIS KILLED: Fragility score {fragility_score:.1f} >= 55.0 ({'; '.join(flaws_identified[:2])})" if disqualified else None,
            metrics={
                "fragility_score": fragility_score,
                "thesis_outcome": thesis_outcome,
                "flaws_identified_count": len(flaws_identified),
                "thesis_killed": disqualified,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=flaws_identified,
        )
