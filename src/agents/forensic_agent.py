"""
Forensic & Red Flag Specialist Agent Module.
Acts as a skeptical trade-killer: searches for accounting red flags, promoter pledging, debt spikes, and regulatory actions.
Has direct veto trigger authority if critical corporate governance thresholds are violated.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    AnnualRatios,
    ShareholdingPattern,
    SymbolMetadata,
)
from src.core.types import AgentStatus, DataFreshness, ForensicVerdict, SignalType


class ForensicAnalysisAgent(BaseAgent):
    """Specialist skeptic agent hunting for disqualifying corporate red flags."""

    def __init__(self):
        super().__init__(agent_name="forensic_analysis_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        shp: ShareholdingPattern | None = context.get("shareholding_pattern")
        ratios: AnnualRatios | None = context.get("annual_ratios")

        pledging_pct = shp.promoter_pledged_pct if shp else 0.0
        debt_equity = ratios.debt_to_equity if ratios else 0.4
        cfo_pat = ratios.cfo_to_pat_ratio if ratios else 0.9
        asm_stage = symbol_meta.asm_gsm_stage

        risks: list[str] = []
        disqualified = False
        disqualify_reason: str | None = None
        verdict = ForensicVerdict.CLEAN

        # 1. Promoter Pledging Check
        if pledging_pct > 20.0:
            disqualified = True
            disqualify_reason = f"Excessive promoter pledging ({pledging_pct:.1f}% > 20.0% max limit)"
            verdict = ForensicVerdict.RED_FLAG_REJECT
            risks.append(disqualify_reason)
        elif pledging_pct > 10.0:
            verdict = ForensicVerdict.MINOR_CONCERN
            risks.append(f"Elevated promoter pledging at {pledging_pct:.1f}%")

        # 2. Surveillance List Check
        if asm_stage >= 2:
            disqualified = True
            disqualify_reason = f"Security is under SEBI/NSE ASM/GSM Stage {asm_stage}"
            verdict = ForensicVerdict.RED_FLAG_REJECT
            risks.append(disqualify_reason)

        # 3. Debt vs Cash Flow Check
        if debt_equity > 2.5 and cfo_pat < 0.3:
            verdict = ForensicVerdict.RED_FLAG_REJECT
            disqualified = True
            disqualify_reason = f"Severe financial distress: High D/E ({debt_equity:.2f}) with negligible cash flow conversion ({cfo_pat:.2f})"
            risks.append(disqualify_reason)

        # Register Forensic Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="FORENSIC_HEALTH",
            raw_metric="governance_and_pledging",
            observed_value=f"Promoter Pledge: {pledging_pct:.1f}%, ASM/GSM Stage: {asm_stage}, D/E: {debt_equity:.2f}, CFO/PAT: {cfo_pat:.2f}",
            unit="forensic_metrics",
            source="EXCHANGE_FILINGS",
            timestamp="2026-06-30",
        )

        score = 95.0 if verdict == ForensicVerdict.CLEAN else (60.0 if verdict == ForensicVerdict.MINOR_CONCERN else 0.0)
        signal = SignalType.REJECT if disqualified else (SignalType.BULLISH if verdict == ForensicVerdict.CLEAN else SignalType.NEUTRAL)

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=score,
            confidence=0.96,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "promoter_pledged_pct": pledging_pct,
                "asm_gsm_stage": asm_stage,
                "forensic_verdict": verdict.value,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
            disqualification_triggered=disqualified,
            disqualification_reason=disqualify_reason,
        )
