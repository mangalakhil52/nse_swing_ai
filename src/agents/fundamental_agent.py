"""
Fundamental Analysis Specialist Agent Module — P0 Integrity Refactored.

Evaluates YoY/QoQ profit acceleration, Free Cash Flow conversion (FCF/PAT, CFO/EBITDA), ROE, ROCE, and Debt/Equity.
ZERO fake fundamental fallbacks. Returns status = AgentStatus.DATA_UNAVAILABLE if required fundamentals are missing.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    AnnualRatios,
    QuarterlyFinancials,
    SymbolMetadata,
)
from src.core.types import AgentStatus, DataFreshness, FundamentalGrade, SignalType


class FundamentalAnalysisAgent(BaseAgent):
    """Specialist agent analyzing earnings acceleration, cash conversion quality, and financial health."""

    def __init__(self):
        super().__init__(agent_name="fundamental_analysis_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        quarterly: list[QuarterlyFinancials] = context.get("quarterly_financials", [])
        ratios: AnnualRatios | None = context.get("annual_ratios")

        # P0 Rule: If fundamental data is unavailable, return DATA_UNAVAILABLE without fabricating values
        if not quarterly and ratios is None:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.NEUTRAL,
                score=0.0,
                confidence=None,
                data_freshness=DataFreshness.RECENT,
                disqualification_triggered=False,
                disqualification_reason=None,
                metrics={"status": "FUNDAMENTALS_UNAVAILABLE"},
                evidence=[],
                risks_identified=["Fundamental financial data unavailable."],
            )

        latest_q = quarterly[0] if quarterly else None
        prev_q = quarterly[1] if len(quarterly) > 1 else None

        sales_growth = latest_q.sales_growth_yoy_pct if latest_q else None
        pat_growth = latest_q.pat_growth_yoy_pct if latest_q else None
        prev_pat_growth = prev_q.pat_growth_yoy_pct if prev_q else None

        earnings_acceleration = (
            round(pat_growth - prev_pat_growth, 1)
            if (pat_growth is not None and prev_pat_growth is not None)
            else None
        )

        roe = ratios.roe_pct if ratios else None
        roce = ratios.roce_pct if ratios else None
        debt_equity = ratios.debt_to_equity if ratios else None
        cfo_pat = ratios.cfo_to_pat_ratio if ratios else None

        fcf_pat_ratio = context.get("fcf_pat_ratio")
        cfo_ebitda_ratio = context.get("cfo_ebitda_ratio")

        # Base Score Computation strictly from observed metrics
        score = 50.0
        metrics_evaluated = 0

        if pat_growth is not None:
            metrics_evaluated += 1
            if earnings_acceleration is not None and earnings_acceleration > 5.0 and pat_growth >= 20.0:
                score += 20.0
            elif pat_growth >= 15.0:
                score += 12.0
            elif pat_growth < 0.0:
                score -= 20.0

        if roe is not None and roce is not None:
            metrics_evaluated += 1
            if roe >= 18.0 and roce >= 20.0:
                score += 15.0
            elif roe >= 12.0:
                score += 8.0

        if fcf_pat_ratio is not None and cfo_ebitda_ratio is not None:
            metrics_evaluated += 1
            if fcf_pat_ratio >= 0.80 and cfo_ebitda_ratio >= 0.70:
                score += 15.0
            elif fcf_pat_ratio < 0.50:
                score -= 15.0

        if debt_equity is not None:
            metrics_evaluated += 1
            if debt_equity <= 0.5:
                score += 10.0
            elif debt_equity > 1.8:
                score -= 25.0

        score = min(100.0, max(0.0, score)) if metrics_evaluated > 0 else 0.0
        signal = (
            SignalType.BULLISH
            if (score >= 70.0 and metrics_evaluated >= 2)
            else (SignalType.BEARISH if (score < 45.0 and metrics_evaluated >= 2) else SignalType.NEUTRAL)
        )

        # Register Evidence strictly for verified observations
        if pat_growth is not None:
            evidence_graph.add_evidence(
                symbol=symbol,
                agent_name=self.agent_name,
                claim_type="EARNINGS_GROWTH",
                raw_metric="pat_growth_yoy_pct",
                observed_value=f"YoY PAT Growth: {pat_growth:.1f}%",
                unit="pct",
                source="QUARTERLY_FINANCIALS",
                timestamp=pd.Timestamp.now().isoformat(),
            )

        if roe is not None:
            evidence_graph.add_evidence(
                symbol=symbol,
                agent_name=self.agent_name,
                claim_type="RETURN_RATIOS",
                raw_metric="roe_pct",
                observed_value=f"ROE: {roe:.1f}%",
                unit="pct",
                source="ANNUAL_RATIOS",
                timestamp=pd.Timestamp.now().isoformat(),
            )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS if metrics_evaluated > 0 else AgentStatus.DATA_UNAVAILABLE,
            signal=signal,
            score=round(score, 1),
            confidence=None,  # Uncalibrated confidence marked as None
            data_freshness=DataFreshness.RECENT,
            disqualification_triggered=False,
            metrics={
                "pat_growth_yoy_pct": pat_growth,
                "earnings_acceleration": earnings_acceleration,
                "roe_pct": roe,
                "roce_pct": roce,
                "debt_to_equity": debt_equity,
                "fcf_to_pat": fcf_pat_ratio,
                "metrics_evaluated_count": metrics_evaluated,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=[],
        )
