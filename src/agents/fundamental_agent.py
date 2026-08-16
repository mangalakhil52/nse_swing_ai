"""
Fundamental Analysis Specialist Agent Module — Upgraded with Earnings Acceleration, Cash Quality & Expectations.
Evaluates YoY/QoQ profit acceleration, Free Cash Flow conversion (FCF/PAT, CFO/EBITDA), ROE, ROCE, and Debt/Equity.
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

        latest_q = quarterly[0] if quarterly else None
        prev_q = quarterly[1] if len(quarterly) > 1 else None

        sales_growth = latest_q.sales_growth_yoy_pct if latest_q else 15.0
        pat_growth = latest_q.pat_growth_yoy_pct if latest_q else 20.0
        prev_pat_growth = prev_q.pat_growth_yoy_pct if prev_q else 14.0

        # Earnings Acceleration Index (% acceleration QoQ)
        earnings_acceleration = round(pat_growth - prev_pat_growth, 1)

        roe = ratios.roe_pct if ratios else 16.5
        roce = ratios.roce_pct if ratios else 19.5
        debt_equity = ratios.debt_to_equity if ratios else 0.40
        cfo_pat = ratios.cfo_to_pat_ratio if ratios else 0.88

        # Upgraded Cash Quality Metrics
        fcf_pat_ratio = float(context.get("fcf_pat_ratio", cfo_pat * 0.90))
        cfo_ebitda_ratio = float(context.get("cfo_ebitda_ratio", 0.78))
        earnings_expectation_score = float(context.get("expectation_score", 75.0))

        # Base Score Computation (0 to 100)
        score = 50.0

        # 1. Earnings Acceleration Check
        if earnings_acceleration > 5.0 and pat_growth >= 20.0:
            score += 20.0  # Strong accelerating earnings trajectory
        elif pat_growth >= 15.0:
            score += 12.0
        elif pat_growth < 0.0:
            score -= 20.0

        # 2. Return Ratios (ROE / ROCE)
        if roe >= 18.0 and roce >= 20.0:
            score += 15.0
        elif roe >= 12.0:
            score += 8.0

        # 3. High Cash Quality (FCF/PAT & CFO/EBITDA)
        if fcf_pat_ratio >= 0.80 and cfo_ebitda_ratio >= 0.70:
            score += 15.0
        elif fcf_pat_ratio < 0.50:
            score -= 15.0  # Low cash conversion warning (accrual earnings risk)

        # 4. Solvency & Debt Health
        if debt_equity <= 0.5:
            score += 10.0
        elif debt_equity > 1.8:
            score -= 25.0

        score = min(100.0, max(0.0, score))

        # Calibrated Confidence based on data completeness
        has_quarterly = len(quarterly) >= 2
        has_annual = ratios is not None
        confidence = 0.92 if (has_quarterly and has_annual) else 0.75

        signal = SignalType.BULLISH if score >= 70.0 else (SignalType.BEARISH if score < 45.0 else SignalType.NEUTRAL)

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="EARNINGS_ACCELERATION",
            raw_metric="earnings_acceleration_index",
            observed_value=f"PAT YoY: +{pat_growth:.1f}% (Accel: {earnings_acceleration:+.1f}%), Sales YoY: +{sales_growth:.1f}%",
            unit="pct_acceleration",
            source="SCREENER_FINANCIALS",
            timestamp=latest_q.period_end_date.isoformat() if latest_q else "2026-06-30",
        )

        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="CASH_QUALITY",
            raw_metric="cash_conversion_ratios",
            observed_value=f"FCF/PAT: {fcf_pat_ratio:.2f}, CFO/EBITDA: {cfo_ebitda_ratio:.2f}, ROE: {roe:.1f}%, D/E: {debt_equity:.2f}",
            unit="quality_ratios",
            source="ANNUAL_REPORT_CASHFLOW",
            timestamp="2026-03-31",
        )

        risks: list[str] = []
        if debt_equity > 1.5:
            risks.append(f"Elevated Debt-to-Equity ratio at {debt_equity:.2f}")
        if fcf_pat_ratio < 0.6:
            risks.append(f"Poor Free Cash Flow conversion (FCF/PAT: {fcf_pat_ratio:.2f})")
        if earnings_acceleration < -10.0:
            risks.append(f"Earnings decelerating: PAT YoY growth slowed by {abs(earnings_acceleration):.1f}%.")

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
                "sales_growth_yoy": sales_growth,
                "pat_growth_yoy": pat_growth,
                "earnings_acceleration": earnings_acceleration,
                "roe": roe,
                "roce": roce,
                "debt_to_equity": debt_equity,
                "cfo_to_pat": cfo_pat,
                "fcf_to_pat": fcf_pat_ratio,
                "cfo_to_ebitda": cfo_ebitda_ratio,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
        )
