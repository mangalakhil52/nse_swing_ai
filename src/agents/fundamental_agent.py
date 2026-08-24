"""Fundamental Analysis Specialist Agent — P0 #14D."""
from datetime import date, datetime
from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, AnnualRatios, QuarterlyFinancials, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SignalType
from src.architecture.contracts import AgentAnalysisResult, StructuredEvidence
from src.data.point_in_time import PointInTimeFilter


class FundamentalAnalysisAgent(BaseAgent):
    """Specialist fundamental desk; no trade construction or final conviction."""

    def __init__(self):
        super().__init__(agent_name="fundamental_analysis_agent")

    async def _analyze(self, symbol_meta: SymbolMetadata, df: pd.DataFrame,
                       evidence_graph: EvidenceGraph, run_id: str,
                       context: dict[str, Any]) -> AgentOutput:
        symbol = symbol_meta.symbol
        raw_quarterly: list[QuarterlyFinancials] = context.get("quarterly_financials", [])
        ratios: AnnualRatios | None = context.get("annual_ratios")
        as_of_date = context.get("as_of_date")
        if as_of_date and raw_quarterly:
            as_of_d = as_of_date if isinstance(as_of_date, date) else pd.to_datetime(as_of_date).date()
            quarterly = PointInTimeFilter.filter_quarterly_financials(raw_quarterly, as_of_d)
        else:
            quarterly = raw_quarterly
        quarterly = sorted(quarterly, key=lambda q: (getattr(q, "available_at", None) or getattr(q, "filing_date", None) or date.min), reverse=True)

        if not quarterly and ratios is None:
            return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE, signal=SignalType.UNKNOWN, score=0.0,
                confidence=None, data_freshness=DataFreshness.UNKNOWN,
                metrics={"status": "FUNDAMENTALS_UNAVAILABLE"}, evidence=[],
                risks_identified=["Fundamental financial data unavailable."])

        latest_q = quarterly[0] if quarterly else None
        prev_q = quarterly[1] if len(quarterly) > 1 else None
        pat_growth = latest_q.pat_growth_yoy_pct if latest_q else None
        prev_pat_growth = prev_q.pat_growth_yoy_pct if prev_q else None
        earnings_acceleration = round(pat_growth - prev_pat_growth, 1) if pat_growth is not None and prev_pat_growth is not None else None
        roe = ratios.roe_pct if ratios else None
        roce = ratios.roce_pct if ratios else None
        debt_equity = ratios.debt_to_equity if ratios else None
        fcf_pat_ratio = context.get("fcf_pat_ratio")
        cfo_ebitda_ratio = context.get("cfo_ebitda_ratio")

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

        score = min(100.0, max(0.0, score)) if metrics_evaluated else 0.0
        signal = SignalType.BULLISH if score >= 70.0 and metrics_evaluated >= 2 else (SignalType.BEARISH if score < 45.0 and metrics_evaluated >= 2 else SignalType.NEUTRAL)

        # EvidenceGraph requires a real timestamp. Do not fabricate one when
        # the source fixture does not provide publication/availability metadata.
        q_timestamp = getattr(latest_q, "available_at", None) or getattr(latest_q, "filing_date", None) if latest_q else None
        if pat_growth is not None and q_timestamp is not None:
            evidence_graph.add_evidence(symbol=symbol, agent_name=self.agent_name, claim_type="EARNINGS_GROWTH",
                raw_metric="pat_growth_yoy_pct", observed_value=f"YoY PAT Growth: {pat_growth:.1f}%", unit="pct",
                source="QUARTERLY_FINANCIALS", timestamp=q_timestamp)
        ratio_timestamp = getattr(ratios, "available_at", None) if ratios else None
        if roe is not None and ratio_timestamp is not None:
            evidence_graph.add_evidence(symbol=symbol, agent_name=self.agent_name, claim_type="RETURN_RATIOS",
                raw_metric="roe_pct", observed_value=f"ROE: {roe:.1f}%", unit="pct", source="ANNUAL_RATIOS",
                timestamp=ratio_timestamp)

        return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id,
            status=AgentStatus.SUCCESS if metrics_evaluated else AgentStatus.DATA_UNAVAILABLE,
            signal=signal, score=round(score, 1), confidence=None, data_freshness=DataFreshness.RECENT,
            metrics={"pat_growth_yoy_pct": pat_growth, "earnings_acceleration": earnings_acceleration,
                     "roe_pct": roe, "roce_pct": roce, "debt_to_equity": debt_equity,
                     "fcf_to_pat": fcf_pat_ratio, "metrics_evaluated_count": metrics_evaluated},
            evidence=evidence_graph.to_evidence_items(symbol), risks_identified=[])

    async def analyze_contract(self, symbol_meta: SymbolMetadata, df: pd.DataFrame,
                               decision_time: datetime | date, run_id: str = "",
                               context: dict[str, Any] | None = None) -> AgentAnalysisResult:
        """Emit #14A AgentAnalysisResult using only fundamentals visible at decision_time."""
        context = dict(context or {})
        context["as_of_date"] = decision_time
        graph = EvidenceGraph()
        output = await self._analyze(symbol_meta, df, graph, run_id, context)
        as_of = decision_time.date() if isinstance(decision_time, datetime) else decision_time
        raw_q = context.get("quarterly_financials", [])
        pit_q = PointInTimeFilter.filter_quarterly_financials(raw_q, as_of) if raw_q else []
        ratios = context.get("annual_ratios")
        pit_ratios = PointInTimeFilter.filter_annual_ratios([ratios], as_of) if ratios is not None else []
        pit_safe = bool(pit_q or pit_ratios) and output.status == AgentStatus.SUCCESS
        signal = output.signal if pit_safe else SignalType.UNKNOWN
        structured = [StructuredEvidence(source="FUNDAMENTAL", observation=item.observed_value,
            as_of=decision_time, direction=signal,
            strength="HIGH" if output.score >= 75 else ("MEDIUM" if output.score >= 55 else "LOW"),
            reliability=1.0 if pit_safe else 0.0, pit_safe=pit_safe) for item in output.evidence]
        return AgentAnalysisResult(symbol=symbol_meta.symbol.upper().strip(), agent_name=self.agent_name,
            decision_time=decision_time, signal=signal, score=output.score if pit_safe else 0.0,
            confidence=0.0, evidence=structured, risks=output.risks_identified, data_quality=None,
            pit_safe=pit_safe, status=output.status if pit_safe else AgentStatus.DATA_UNAVAILABLE,
            reasons=output.risks_identified if pit_safe else ["FUNDAMENTALS_UNAVAILABLE_OR_PIT_UNVERIFIED"])
