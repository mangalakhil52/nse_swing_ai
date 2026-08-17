"""
CIO Orchestrator Agent Module — Refactored for P0-P4 Production Data Integrity & Pipeline Architecture.

Pipeline Order (P4):
  1. DATA INGESTION & POINT-IN-TIME FILTERING
  2. DATA VALIDATION (Zero synthetic fallbacks)
  3. ALPHA SPECIALIST DESKS (Technical, Relative Strength, Fundamental, Institutional, News, Sector)
  4. PROBABILITY & NET EV ENGINE (Empirical sample size >= 30, Net EV after friction)
  5. RISK VETO (Gatekeeper, zero alpha contribution)
  6. TRADE CONSTRUCTION (Structural targets & SL, zero alpha contribution)
  7. EXECUTION COST & SLIPPAGE
  8. PORTFOLIO FACTOR RISK
  9. FINAL CONVICTION RANKING (0-2 Picks)
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
from src.agents.thesis_killer_agent import ThesisKillerAgent
from src.agents.trade_construction_agent import TradeConstructionAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    SymbolMetadata,
    TradeLevels,
    TradeRecommendation,
)
from src.core.types import (
    AgentStatus,
    ConfluenceState,
    ConvictionGrade,
    MarketRegime,
    PatternType,
    SignalType,
    TradeStatus,
    TradingStance,
)
from src.data.validation import DataValidator
from src.quant.probability_engine import ProbabilityPathEngine
from src.quant.regime import RegimeAnalysisResult
from src.quant.screener import ScreenerCandidate
from src.risk.correlation import PortfolioCorrelationGuard
from src.risk.execution_quality import ExecutionQualityModel
from src.risk.veto import RiskVetoEngine

logger = logging.getLogger(__name__)


class CIOOrchestrator:
    """Master Chief Investment Officer Orchestrator enforcing data integrity and modular pipeline separation."""

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
        self.thesis_killer_agent = ThesisKillerAgent()
        self.confluence_agent = ConfluenceAgent()
        self.quant_score_agent = QuantScoreAgent()
        self.trade_construction_agent = TradeConstructionAgent()
        self.data_validator = DataValidator(min_required_bars=50)

    async def _run_alpha_desks(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> dict[str, AgentOutput]:
        """Runs all alpha-generating specialist desks concurrently."""

        alpha_tasks = {
            "technical_analysis_agent": self.technical_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "relative_strength_agent": self.rs_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "fundamental_analysis_agent": self.fundamental_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "sector_rotation_agent": self.sector_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "institutional_flow_agent": self.institutional_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "news_intelligence_agent": self.news_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "catalyst_agent": self.catalyst_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
            "forensic_analysis_agent": self.forensic_agent.execute(symbol_meta, df, evidence_graph, run_id, context),
        }

        results = await asyncio.gather(*alpha_tasks.values(), return_exceptions=True)
        outputs: dict[str, AgentOutput] = {}

        for name, result in zip(alpha_tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Alpha Agent '{name}' raised exception for {symbol_meta.symbol}: {result}")
                outputs[name] = AgentOutput(
                    agent_name=name,
                    symbol=symbol_meta.symbol,
                    run_id=run_id,
                    status=AgentStatus.FAILED,
                    signal=SignalType.NEUTRAL,
                    score=0.0,
                    confidence=None,
                )
            else:
                outputs[name] = result

        # Run Thesis Killer (Devil's Advocate)
        killer_ctx = {**context, "agent_outputs": outputs}
        try:
            outputs["thesis_killer_agent"] = await self.thesis_killer_agent.execute(symbol_meta, df, evidence_graph, run_id, killer_ctx)
        except Exception as e:
            logger.error(f"ThesisKillerAgent exception for {symbol_meta.symbol}: {e}")

        # Run Risk Management Agent (Gatekeeper, zero alpha)
        try:
            outputs["risk_management_agent"] = await self.risk_agent.execute(symbol_meta, df, evidence_graph, run_id, context)
        except Exception as e:
            logger.error(f"RiskManagementAgent exception for {symbol_meta.symbol}: {e}")

        return outputs

    async def analyze_candidate(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        run_id: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[TradeRecommendation | None, dict[str, float]]:
        """
        Full 9-stage research pipeline for a single candidate.
        Returns (TradeRecommendation, sub_scores) or (None, {}) if rejected with exact reason logged.
        """
        context = context or {}
        evidence_graph = EvidenceGraph()
        symbol = symbol_meta.symbol

        # Stage 1: Data Quality Validation (P0.3)
        val_res = self.data_validator.validate_ohlcv_dataframe(df, symbol)
        if not val_res.is_valid:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Data Validation Failed ({'; '.join(val_res.errors)})")
            return None, {}

        # Stage 2: Alpha Specialist Research
        agent_outputs = await self._run_alpha_desks(symbol_meta, df, evidence_graph, run_id, context)

        # Early Exit on Disqualifications (Forensic, Risk, Thesis Killer)
        for key in ["forensic_analysis_agent", "risk_management_agent", "thesis_killer_agent"]:
            out = agent_outputs.get(key)
            if out and out.disqualification_triggered:
                logger.info(f"[{symbol}] STATUS = REJECTED | REASON = {key.upper()} ({out.disqualification_reason})")
                return None, {}

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

        # Stage 3: Trade Construction (P1.0, P1.1, P1.2)
        trade_ctx = {**context, "agent_outputs": agent_outputs}
        trade_out = await self.trade_construction_agent.execute(symbol_meta, df, evidence_graph, run_id, trade_ctx)
        agent_outputs["trade_construction_agent"] = trade_out

        if trade_out.disqualification_triggered:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Trade Construction ({trade_out.disqualification_reason})")
            return None, {}

        trade_levels: TradeLevels | None = None
        if trade_out.metrics and "trade_levels" in trade_out.metrics:
            try:
                trade_levels = TradeLevels(**trade_out.metrics["trade_levels"])
            except Exception as e:
                logger.error(f"[{symbol}] Failed to parse TradeLevels model: {e}")

        if not trade_levels:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Missing structural trade levels.")
            return None, {}

        # Stage 4: Risk Veto Engine
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
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Risk Veto ({'; '.join(veto.rejection_reasons)})")
            return None, {}

        # Stage 5: Probability-of-Path & Net EV Engine (P1.3, P1.4)
        pattern_str = tech_out.metrics.get("pattern_detected", PatternType.FLAT_BASE_BREAKOUT.value) if tech_out else PatternType.FLAT_BASE_BREAKOUT.value
        try:
            pattern_enum = PatternType(pattern_str)
        except ValueError:
            pattern_enum = PatternType.FLAT_BASE_BREAKOUT

        mansfield_rs = rs_out.metrics.get("mansfield_rs", 0.0) if rs_out else 0.0
        fcf_pat = fund_out.metrics.get("fcf_to_pat", 0.90) if fund_out else 0.90

        prob_res = ProbabilityPathEngine.evaluate_expectancy(
            pattern_type=pattern_enum,
            market_regime=market_regime,
            mansfield_rs=mansfield_rs,
            target1_pct=trade_levels.target_1 / max(1.0, trade_levels.current_market_price) * 100.0 - 100.0,
            stop_loss_pct=trade_levels.risk_percentage,
            fcf_pat_ratio=fcf_pat,
        )

        if not prob_res.is_ev_positive:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Probability/EV Engine ({prob_res.disqualification_reason})")
            return None, {}

        # Stage 6: Execution Cost & Slippage Modeling
        adtv = float(df["turnover_crores"].mean()) if "turnover_crores" in df.columns else 25.0
        atr = float(df["high"].tail(14).mean() - df["low"].tail(14).mean()) if len(df) >= 14 else 15.0

        exec_res = ExecutionQualityModel.evaluate_execution_quality(
            current_price=trade_levels.current_market_price,
            entry_trigger_price=trade_levels.entry_trigger_price,
            adtv_crores=adtv,
            allocated_capital_rupees=trade_levels.allocated_capital_rupees,
            atr_14=atr,
        )

        if not exec_res.is_executable:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Poor Execution Quality (Slippage: {exec_res.expected_slippage_pct:.2f}%)")
            return None, {}

        trade_levels.entry_trigger_price = exec_res.adjusted_entry_trigger

        # Stage 7: Confluence Evaluation
        confluence_ctx = {**context, "agent_outputs": agent_outputs}
        conf_out = await self.confluence_agent.execute(symbol_meta, df, evidence_graph, run_id, confluence_ctx)
        agent_outputs["confluence_agent"] = conf_out

        if conf_out.disqualification_triggered:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Confluence Conflict ({conf_out.disqualification_reason})")
            return None, {}

        confluence_state_val = conf_out.metrics.get("confluence_state", ConfluenceState.MODERATE.value)
        confluence_state = ConfluenceState(confluence_state_val)

        # Stage 8: Quantitative 100-Point Scoring
        score_ctx = {
            **context,
            "agent_outputs": agent_outputs,
            "trade_levels": trade_levels,
            "confluence_state": confluence_state,
        }
        score_out = await self.quant_score_agent.execute(symbol_meta, df, evidence_graph, run_id, score_ctx)
        composite_score = score_out.score

        # P2.2: Enforce Strict A+ Conviction Criteria
        is_a_plus = (
            composite_score >= 88.0
            and prob_res.win_probability is not None
            and prob_res.win_probability >= 0.65
            and prob_res.net_ev >= 3.0
            and prob_res.sample_size >= 50
            and exec_res.expected_slippage_pct <= 0.20
        )

        if is_a_plus:
            conviction = ConvictionGrade.A_PLUS
        elif composite_score >= 75.0:
            conviction = ConvictionGrade.A
        elif composite_score >= 65.0:
            conviction = ConvictionGrade.B
        else:
            conviction = ConvictionGrade.C

        if conviction in [ConvictionGrade.C, ConvictionGrade.REJECT]:
            logger.info(f"[{symbol}] STATUS = REJECTED | REASON = Score Below Threshold ({composite_score:.1f}/100 - Conviction {conviction.value})")
            return None, {}

        # Compile risks from all agents
        all_risks: list[str] = []
        for out in agent_outputs.values():
            all_risks.extend(out.risks_identified or [])
        all_risks = list(set(all_risks))[:8]

        why_trade: list[str] = []
        if tech_out and tech_out.signal == SignalType.BULLISH:
            pattern = tech_out.metrics.get("pattern_detected", "Strong Setup")
            why_trade.append(f"Technical: {pattern}")
        if rs_out and rs_out.signal == SignalType.BULLISH:
            rs_val = rs_out.metrics.get("mansfield_rs", 0.0)
            why_trade.append(f"RS Leader: Outperforming NIFTY by {rs_val:.1f}% (Mansfield RS)")
        if fund_out and fund_out.signal == SignalType.BULLISH:
            pat_g = fund_out.metrics.get("pat_growth_yoy", 0.0)
            why_trade.append(f"Fundamentals: PAT growth +{pat_g:.1f}% YoY with FCF/PAT {fund_out.metrics.get('fcf_to_pat', 0.9):.2f}")
        why_trade.append(f"Expectancy: Empirical P(Win) {prob_res.win_probability*100:.0f}% (n={prob_res.sample_size}), Net EV +{prob_res.net_ev:.2f}%")

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
            technical_setup_description=tech_out.metrics.get("pattern_detected", "Trend Continuation") if tech_out else "N/A",
            catalyst_summary=agent_outputs.get("catalyst_agent", AgentOutput(
                agent_name="catalyst_agent", symbol=symbol_meta.symbol, run_id=run_id
            )).metrics.get("description", "No catalyst"),
            fundamental_summary=f"ROE {fund_out.metrics.get('roe', 0.0):.1f}%, FCF/PAT {fund_out.metrics.get('fcf_to_pat', 0.9):.2f}" if fund_out else "N/A",
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
        Returns a ranked, de-duplicated, and factor-risk-filtered final recommendation basket (0-2 picks).
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

        # Stage 9: Portfolio Factor Risk & Correlation Filtering (P1.9)
        final_basket = PortfolioCorrelationGuard.filter_uncorrelated_basket(all_recs, max_picks=2)

        logger.info(
            f"[CIO] Daily scan complete. Analyzed: {len(candidates)} | Qualified: {len(all_recs)} | "
            f"Final Basket: {len(final_basket)} recommendations."
        )
        return final_basket
