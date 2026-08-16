"""
Quantitative Scoring Engine Agent Module.
Calculates transparent 100-point composite score with calibrated multi-factor weights.
Applies conviction grading (A+, A, B+, B, C, REJECT).
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    CandidateScore,
    SymbolMetadata,
    TradeLevels,
)
from src.core.types import (
    AgentStatus,
    ConfluenceState,
    ConvictionGrade,
    DataFreshness,
    MarketRegime,
    SignalType,
)


class QuantScoreAgent(BaseAgent):
    """Calculates the 100-point composite factor score for shortlisted candidates."""

    def __init__(self):
        super().__init__(agent_name="quant_score_agent")

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
        trade_levels: TradeLevels | None = context.get("trade_levels")
        market_regime: MarketRegime = context.get("market_regime", MarketRegime.BULL)
        confluence_state: ConfluenceState = context.get("confluence_state", ConfluenceState.MODERATE)

        # 1. Component Sub-Scores Calculation (Weighted out of 100)
        # Factor 1: Technical Setup (Max 20 pts)
        tech_out = agent_outputs.get("technical_analysis_agent")
        tech_score = (tech_out.score / 100.0 * 20.0) if tech_out else 10.0

        # Factor 2: Relative Strength (Max 15 pts)
        rs_out = agent_outputs.get("relative_strength_agent")
        rs_score = (rs_out.score / 100.0 * 15.0) if rs_out else 8.0

        # Factor 3: Risk / Reward Geometry (Max 15 pts)
        rr_score = 10.0
        if trade_levels:
            if trade_levels.risk_reward_t1 >= 2.0:
                rr_score = 15.0
            elif trade_levels.risk_reward_t1 >= 1.8:
                rr_score = 12.0
            else:
                rr_score = 6.0

        # Factor 4: Market Regime (Max 10 pts)
        regime_weights = {
            MarketRegime.STRONG_BULL: 10.0,
            MarketRegime.BULL: 8.0,
            MarketRegime.NEUTRAL: 5.0,
            MarketRegime.BEAR: 2.0,
            MarketRegime.STRONG_BEAR: 0.0,
        }
        regime_score = regime_weights.get(market_regime, 5.0)

        # Factor 5: Volume & Delivery (Max 10 pts)
        inst_out = agent_outputs.get("institutional_flow_agent")
        vol_score = (inst_out.score / 100.0 * 10.0) if inst_out else 5.0

        # Factor 6: Momentum (Max 10 pts)
        mom_score = 7.5
        if not df.empty and "rsi_14" in df.columns:
            rsi = df["rsi_14"].iloc[-1]
            if 58.0 <= rsi <= 72.0:
                mom_score = 10.0
            elif 50.0 <= rsi < 58.0:
                mom_score = 7.0
            else:
                mom_score = 4.0

        # Factor 7: Fundamental Quality (Max 10 pts)
        fund_out = agent_outputs.get("fundamental_analysis_agent")
        fund_score = (fund_out.score / 100.0 * 10.0) if fund_out else 5.0

        # Factor 8: Catalyst / News (Max 5 pts)
        news_out = agent_outputs.get("news_intelligence_agent")
        cat_out = agent_outputs.get("catalyst_agent")
        news_score = (news_out.score / 100.0 * 2.5 if news_out else 1.5) + (cat_out.score / 100.0 * 2.5 if cat_out else 1.5)

        # Factor 9: Sector Strength (Max 5 pts)
        sec_out = agent_outputs.get("sector_rotation_agent")
        sec_score = (sec_out.score / 100.0 * 5.0) if sec_out else 2.5

        # 2. Total Composite Score
        total_score = (
            tech_score
            + rs_score
            + rr_score
            + regime_score
            + vol_score
            + mom_score
            + fund_score
            + news_score
            + sec_score
        )
        total_score = round(min(100.0, max(0.0, total_score)), 1)

        # 3. Conviction Grading
        if confluence_state == ConfluenceState.CONFLICTED:
            conviction = ConvictionGrade.REJECT
        elif total_score >= 88.0 and confluence_state == ConfluenceState.VERY_HIGH:
            conviction = ConvictionGrade.A_PLUS
        elif total_score >= 80.0:
            conviction = ConvictionGrade.A
        elif total_score >= 72.0:
            conviction = ConvictionGrade.B_PLUS
        elif total_score >= 60.0:
            conviction = ConvictionGrade.B
        else:
            conviction = ConvictionGrade.C

        factor_breakdown = {
            "technical_setup": round(tech_score, 1),
            "relative_strength": round(rs_score, 1),
            "risk_reward": round(rr_score, 1),
            "market_regime": round(regime_score, 1),
            "volume_delivery": round(vol_score, 1),
            "momentum": round(mom_score, 1),
            "fundamental_quality": round(fund_score, 1),
            "catalyst_news": round(news_score, 1),
            "sector_strength": round(sec_score, 1),
        }

        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="QUANT_SCORE",
            raw_metric="composite_score_100",
            observed_value=f"Score: {total_score}/100 -> Conviction Grade: {conviction.value}",
            unit="points",
            source="SCORING_ENGINE",
            timestamp="EOD",
        )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=SignalType.BULLISH if conviction in [ConvictionGrade.A_PLUS, ConvictionGrade.A] else SignalType.NEUTRAL,
            score=total_score,
            confidence=0.95,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "composite_score": total_score,
                "conviction_grade": conviction.value,
                "factor_scores": factor_breakdown,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=[],
        )
