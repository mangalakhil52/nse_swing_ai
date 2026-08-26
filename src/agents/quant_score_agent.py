"""
Quantitative Scoring Engine Agent Module.
Calculates a transparent 100-point composite score from observed specialist outputs.
Missing evidence contributes zero; it is never replaced by optimistic defaults.
"""

from datetime import datetime
from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, CandidateScore, SymbolMetadata, TradeLevels
from src.core.types import AgentStatus, ConfluenceState, ConvictionGrade, DataFreshness, MarketRegime, SignalType


class QuantScoreAgent(BaseAgent):
    """Calculates the 100-point composite factor score for shortlisted candidates."""

    def __init__(self):
        super().__init__(agent_name="quant_score_agent")

    @staticmethod
    def _weighted(output: AgentOutput | None, weight: float) -> float:
        if output is None or output.status in {AgentStatus.DATA_UNAVAILABLE, AgentStatus.FAILED}:
            return 0.0
        return max(0.0, min(100.0, float(output.score))) / 100.0 * weight

    async def _analyze(self, symbol_meta: SymbolMetadata, df: pd.DataFrame, evidence_graph: EvidenceGraph,
                       run_id: str, context: dict[str, Any]) -> AgentOutput:
        symbol = symbol_meta.symbol
        agent_outputs: dict[str, AgentOutput] = context.get("agent_outputs", {})
        trade_levels: TradeLevels | None = context.get("trade_levels")
        market_regime: MarketRegime = context.get("market_regime", MarketRegime.UNKNOWN)
        confluence_state: ConfluenceState = context.get("confluence_state", ConfluenceState.MODERATE)

        tech_out = agent_outputs.get("technical_analysis_agent")
        rs_out = agent_outputs.get("relative_strength_agent")
        inst_out = agent_outputs.get("institutional_flow_agent")
        fund_out = agent_outputs.get("fundamental_analysis_agent")
        news_out = agent_outputs.get("news_intelligence_agent")
        cat_out = agent_outputs.get("catalyst_agent")
        sec_out = agent_outputs.get("sector_rotation_agent")

        tech_score = self._weighted(tech_out, 20.0)
        rs_score = self._weighted(rs_out, 15.0)

        rr_score = 0.0
        if trade_levels:
            if trade_levels.risk_reward_t1 >= 2.0:
                rr_score = 15.0
            elif trade_levels.risk_reward_t1 >= 1.8:
                rr_score = 12.0
            elif trade_levels.risk_reward_t1 >= 1.5:
                rr_score = 8.0

        regime_score = {
            MarketRegime.STRONG_BULL: 10.0,
            MarketRegime.BULL: 8.0,
            MarketRegime.NEUTRAL: 5.0,
            MarketRegime.BEAR: 2.0,
            MarketRegime.STRONG_BEAR: 0.0,
        }.get(market_regime, 0.0)

        vol_score = self._weighted(inst_out, 10.0)
        mom_score = 0.0
        if not df.empty and "rsi_14" in df.columns and pd.notna(df["rsi_14"].iloc[-1]):
            rsi = float(df["rsi_14"].iloc[-1])
            if 58.0 <= rsi <= 72.0:
                mom_score = 10.0
            elif 50.0 <= rsi < 58.0:
                mom_score = 7.0
            elif 45.0 <= rsi < 50.0 or 72.0 < rsi <= 78.0:
                mom_score = 4.0

        fund_score = self._weighted(fund_out, 10.0)
        news_score = self._weighted(news_out, 2.5) + self._weighted(cat_out, 2.5)
        sec_score = self._weighted(sec_out, 5.0)

        total_score = round(min(100.0, max(0.0, tech_score + rs_score + rr_score + regime_score + vol_score + mom_score + fund_score + news_score + sec_score)), 1)

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
            "technical_setup": round(tech_score, 1), "relative_strength": round(rs_score, 1),
            "risk_reward": round(rr_score, 1), "market_regime": round(regime_score, 1),
            "volume_delivery": round(vol_score, 1), "momentum": round(mom_score, 1),
            "fundamental_quality": round(fund_score, 1), "catalyst_news": round(news_score, 1),
            "sector_strength": round(sec_score, 1),
        }
        observed_at = context.get("decision_time") or (df["timestamp"].iloc[-1] if "timestamp" in df.columns and not df.empty else None)
        if observed_at is not None:
            evidence_graph.add_evidence(symbol=symbol, agent_name=self.agent_name, claim_type="QUANT_SCORE",
                raw_metric="composite_score_100", observed_value=f"Score: {total_score}/100 -> {conviction.value}",
                unit="points", source="SCORING_ENGINE", timestamp=observed_at)

        return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id, status=AgentStatus.SUCCESS,
            signal=SignalType.BULLISH if conviction in [ConvictionGrade.A_PLUS, ConvictionGrade.A] else SignalType.NEUTRAL,
            score=total_score, confidence=None if total_score == 0 else 0.95, data_freshness=DataFreshness.RECENT,
            metrics={"composite_score": total_score, "conviction_grade": conviction.value, "factor_scores": factor_breakdown},
            evidence=evidence_graph.to_evidence_items(symbol), risks_identified=[])
