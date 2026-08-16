"""
CIO Orchestrator Agent Module.
Master coordinator that runs all specialist agents in parallel for each shortlisted candidate,
collects outputs, applies the 100-pt scoring model, enforces risk vetoes, constructs trade levels,
and generates final ranked TradeRecommendation dossiers.
"""

import asyncio
from datetime import date, datetime
import logging
import uuid
from typing import Any
import pandas as pd

from config.settings import settings
from src.agents.catalyst_agent import CatalystAgent
from src.agents.confluence_agent import ConfluenceAgent
from src.agents.forensic_agent import ForensicAnalysisAgent
from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.agents.institutional_agent import InstitutionalFlowAgent
from src.agents.news_agent import NewsIntelligenceAgent
from src.agents.quant_score_agent import QuantScoreAgent
from src.agents.relative_strength_agent import RelativeStrengthAgent
from src.agents.risk_agent import RiskManagementAgent
from src.agents.sector_agent import SectorRotationAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.trade_construction_agent import TradeConstructionAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    CandidateScore,
    SymbolMetadata,
    TradeLevels,
    TradeRecommendation,
)
from src.core.types import (
    AgentStatus,
    ConfluenceState,
    ConvictionGrade,
    MarketRegime,
    SignalType,
    TradeStatus,
    TradingStance,
)
from src.quant.regime import RegimeAnalysisResult
from src.quant.screener import ScreenerCandidate
from src.risk.correlation import PortfolioCorrelationGuard
from src.risk.veto import RiskVetoEngine

logger = logging.getLogger(__name__)


class CIOOrchestrator:
    """
    Chief Investment Officer Orchestrator.
    Runs all domain specialist agents in parallel, evaluates scoring, applies vetoes,
    and surfaces top 0–2 actionable trade recommendations per daily scan cycle.
    """

    def __init__(self):
        self.technical_agent = TechnicalAnalysisAgent()
        self.rs_agent = RelativeStrengthAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.sector_agent = SectorRotationAgent()
        self.institutional_agent = InstitutionalFlowAgent()
        self.news_agent = NewsIntelligenceAgent()
        self.catalyst_agent = CatalystAgent()
        self.forensic_agent = ForensicAnalysisAgent()
        self.risk_agent = RiskManagementAgent()
        self.confluence_agent = ConfluenceAgent()
        self.quant_score_agent = QuantScoreAgent()
        self.trade_construction_agent = TradeConstructionAgent()

    async def _run_parallel_agents(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> dict[str, AgentOutput]:
        """Runs all tier-1 domain specialist agents concurrently."""

        tier1_tasks = {
            "technical_analysis_agent": self.technical_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "relative_strength_agent": self.rs_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "fundamental_analysis_agent": self.fundamental_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "sector_rotation_agent": self.sector_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "institutional_flow_agent": self.institutional_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "news_intelligence_agent": self.news_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "catalyst_agent": self.catalyst_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "forensic_analysis_agent": self.forensic_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "risk_management_agent": self.risk_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
        }

        results = await asyncio.gather(*tier1_tasks.values(), return_exceptions=True)
        outputs: dict[str, AgentOutput] = {}

        for name, result in zip(tier1_tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Agent '{name}' raised exception for {symbol_meta.symbol}: {result}")
                outputs[name] = AgentOutput(
                    agent_name=name,
                    symbol=symbol_meta.symbol,
                    run_id=run_id,
                    status=AgentStatus.FAILED,
                    signal=SignalType.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                )
            else:
                outputs[name] = result

        return outputs

    async def analyze_candidate(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        run_id: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[TradeRecommendation | None, dict[str, float]]:
        """
        Full research pipeline for a single shortlisted candidate.
        Returns (TradeRecommendation, sub_scores) or (None, {}) if disqualified.
        """
        context = context or {}
        evidence_graph = EvidenceGraph()

        # Phase 1: Parallel specialist research
        agent_outputs = await self._run_parallel_agents(symbol_meta, df, evidence_graph, run_id, context)

        # Early exit if forensic or risk agent disqualified
        for key in ["forensic_analysis_agent", "risk_management_agent"]:
            out = agent_outputs.get(key)
            if out and out.disqualification_triggered:
                logger.info(f"[{symbol_meta.symbol}] DISQUALIFIED by {key}: {out.disqualification_reason}")
                return None, {}

        # Extract individual desk quality scores for factor-based tie-breaking
        rs_out = agent_outputs.get("relative_strength_agent")
        tech_out = agent_outputs.get("technical_analysis_agent")
        fund_out = agent_outputs.get("fundamental_analysis_agent")
        inst_out = agent_outputs.get("institutional_flow_agent")

        desk_sub_scores = {
            "rs_score": rs_out.score if rs_out else 0.0,
            "tech_score": tech_out.score if tech_out else 0.0,
            "fund_score": fund_out.score if fund_out else 0.0,
            "inst_score": inst_out.score if inst_out else 0.0,
        }

        # Phase 2: Trade geometry construction
        trade_ctx = {**context, "agent_outputs": agent_outputs}
        trade_out = await self.trade_construction_agent.execute(symbol_meta, df, evidence_graph, run_id, trade_ctx)
        agent_outputs["trade_construction_agent"] = trade_out

        trade_levels: TradeLevels | None = None
        if trade_out.metrics and "trade_levels" in trade_out.metrics:
            try:
                trade_levels = TradeLevels(**trade_out.metrics["trade_levels"])
            except Exception:
                pass

        # Phase 3: Risk veto evaluation
        market_regime: MarketRegime = context.get("market_regime", MarketRegime.BULL)
        trading_stance: TradingStance = context.get("trading_stance", TradingStance.NORMAL)

        veto = RiskVetoEngine.evaluate_candidate(
            symbol_meta=symbol_meta,
            agent_outputs=agent_outputs,
            trade_levels=trade_levels,
            market_regime=market_regime,
            trading_stance=trading_stance,
        )

        if not veto.passed:
            logger.info(f"[{symbol_meta.symbol}] RISK VETO: {veto.rejection_reasons}")
            return None, {}

        # Phase 4: Confluence evaluation
        confluence_ctx = {**context, "agent_outputs": agent_outputs}
        conf_out = await self.confluence_agent.execute(symbol_meta, df, evidence_graph, run_id, confluence_ctx)
        agent_outputs["confluence_agent"] = conf_out

        if conf_out.disqualification_triggered:
            logger.info(f"[{symbol_meta.symbol}] CONFLUENCE CONFLICT: {conf_out.disqualification_reason}")
            return None, {}

        confluence_state_val = conf_out.metrics.get("confluence_state", ConfluenceState.MODERATE.value)
        confluence_state = ConfluenceState(confluence_state_val)

        # Phase 5: 100-point scoring
        score_ctx = {
            **context,
            "agent_outputs": agent_outputs,
            "trade_levels": trade_levels,
            "confluence_state": confluence_state,
        }
        score_out = await self.quant_score_agent.execute(symbol_meta, df, evidence_graph, run_id, score_ctx)
        composite_score = score_out.score
        conviction_str = score_out.metrics.get("conviction_grade", ConvictionGrade.B.value)
        conviction = ConvictionGrade(conviction_str)

        # Reject low-conviction candidates (must be B+ or higher: >= 72 score)
        if conviction in [ConvictionGrade.C, ConvictionGrade.REJECT]:
            logger.info(f"[{symbol_meta.symbol}] Below conviction threshold: {conviction.value} ({composite_score:.1f})")
            return None, {}

        # Compile risks from all agents
        all_risks: list[str] = []
        for out in agent_outputs.values():
            all_risks.extend(out.risks_identified or [])
        all_risks = list(set(all_risks))[:8]

        # Why this trade bullets
        why_trade: list[str] = []
        tech = agent_outputs.get("technical_analysis_agent")
        rs = agent_outputs.get("relative_strength_agent")
        fund = agent_outputs.get("fundamental_analysis_agent")
        if tech and tech.signal == SignalType.BULLISH:
            pattern = tech.metrics.get("pattern_detected", "Strong Chart Setup")
            why_trade.append(f"Technical: {pattern}")
        if rs and rs.signal == SignalType.BULLISH:
            rs_val = rs.metrics.get("mansfield_rs", 0.0)
            why_trade.append(f"RS Leader: Outperforming NIFTY by {rs_val:.1f}% (Mansfield RS)")
        if fund and fund.signal == SignalType.BULLISH:
            pat_g = fund.metrics.get("pat_growth_yoy", 0.0)
            why_trade.append(f"Fundamentals: PAT growth +{pat_g:.1f}% YoY with improving return ratios")

        if not trade_levels:
            logger.warning(f"[{symbol_meta.symbol}] DISQUALIFIED: Trade construction agent failed to build explicit ATR/pattern-based levels.")
            return None, {}

        rec_id = f"REC-{datetime.utcnow().strftime('%Y%m%d')}-{symbol_meta.symbol}-{uuid.uuid4().hex[:6].upper()}"

        recommendation = TradeRecommendation(
            recommendation_id=rec_id,
            run_id=run_id,
            symbol=symbol_meta.symbol,
            company_name=symbol_meta.company_name,
            sector=symbol_meta.sector or "General",
            recommendation_date=date.today(),
            conviction=conviction,
            composite_score=composite_score,
            levels=trade_levels,
            technical_setup_description=tech.metrics.get("pattern_detected", "Trend Continuation") if tech else "N/A",
            catalyst_summary=agent_outputs.get("catalyst_agent", AgentOutput(
                agent_name="catalyst_agent", symbol=symbol_meta.symbol, run_id=run_id
            )).metrics.get("description", "No catalyst"),
            fundamental_summary=f"ROE {fund.metrics.get('roe', 0.0):.1f}%, D/E {fund.metrics.get('debt_to_equity', 0.0):.2f}" if fund else "N/A",
            sector_context=f"Sector rank: #{agent_outputs.get('sector_rotation_agent', AgentOutput(agent_name='s', symbol=symbol_meta.symbol, run_id=run_id)).metrics.get('sector_rank', '?')}",
            market_regime=market_regime.value,
            major_risks=all_risks,
            invalidation_rules=trade_levels.invalidation_criteria,
            why_this_trade=why_trade,
            evidence_dossier=evidence_graph.to_evidence_items(symbol_meta.symbol),
            status=TradeStatus.PENDING_ENTRY,
        )

        return recommendation, desk_sub_scores

    async def run_daily_scan(
        self,
        candidates: list[ScreenerCandidate],
        stock_dfs: dict[str, pd.DataFrame],
        universe: dict[str, SymbolMetadata],
        regime_result: RegimeAnalysisResult,
        run_id: str,
        shared_context: dict[str, Any] | None = None,
    ) -> list[TradeRecommendation]:
        """
        Runs the complete CIO research pipeline across all Stage-1 shortlisted candidates.
        Returns a ranked, de-duplicated, and correlation-filtered final recommendation basket.
        """
        logger.info(f"[CIO] Starting daily scan for {len(candidates)} candidates. Run ID: {run_id}")
        logger.info(f"[CIO] Market Regime: {regime_result.regime.value} | Stance: {regime_result.trading_stance.value}")

        if not regime_result.allow_long_swing_trades:
            logger.warning("[CIO] Market regime prohibits long swing trades. Scan aborted.")
            return []

        base_context = {
            "market_regime": regime_result.regime,
            "trading_stance": regime_result.trading_stance,
            "regime_risk_multiplier": regime_result.risk_multiplier,
            **(shared_context or {}),
        }

        candidate_evaluations: list[tuple[TradeRecommendation, dict[str, float]]] = []

        for cand in candidates:
            symbol = cand.symbol
            df = stock_dfs.get(symbol)
            sym_meta = universe.get(symbol)

            if df is None or sym_meta is None:
                continue

            enriched_df = cand.enriched_df if cand.enriched_df is not None else df

            try:
                rec, sub_scores = await self.analyze_candidate(sym_meta, enriched_df, run_id, base_context)
                if rec:
                    candidate_evaluations.append((rec, sub_scores))
            except Exception as e:
                logger.error(f"[CIO] Error analyzing {symbol}: {e}", exc_info=True)

        # STRICT QUANTITATIVE MULTI-FACTOR SORTING (ZERO ALPHABETICAL BIAS)
        # Factor 1: Composite 100-pt Score (Descending)
        # Factor 2: Relative Strength Outperformance vs Nifty (Descending)
        # Factor 3: Technical Pattern & Volume Surge Score (Descending)
        # Factor 4: Fundamentals PAT growth & ROE Score (Descending)
        # Factor 5: Delivery Buying & Institutional Flow Score (Descending)
        candidate_evaluations.sort(
            key=lambda item: (
                -item[0].composite_score,
                -item[1].get("rs_score", 0.0),
                -item[1].get("tech_score", 0.0),
                -item[1].get("fund_score", 0.0),
                -item[1].get("inst_score", 0.0),
            )
        )

        all_recs = [item[0] for item in candidate_evaluations]

        # Apply portfolio correlation guard (max 2 picks, max 1 per sector)
        final_basket = PortfolioCorrelationGuard.filter_uncorrelated_basket(all_recs, max_picks=2)

        logger.info(
            f"[CIO] Daily scan complete. Analyzed: {len(candidates)} | Qualified: {len(all_recs)} | "
            f"Final Basket: {len(final_basket)} recommendations."
        )
        return final_basket
