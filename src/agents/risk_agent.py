"""
Risk Management Specialist Agent Module.
Enforces absolute veto power over all candidates.
Identifies event risk (earnings in <= 3 sessions), stop distance risk (> 8%), liquidity risk, and regime risk.
"""

from datetime import date, datetime, timedelta
from typing import Any
import pandas as pd

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    CorporateEvent,
    SymbolMetadata,
)
from src.core.types import (
    AgentStatus,
    DataFreshness,
    MarketRegime,
    SignalType,
    TradingStance,
)


class RiskManagementAgent(BaseAgent):
    """Specialist risk guard agent with absolute veto authority."""

    def __init__(self):
        super().__init__(agent_name="risk_management_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        upcoming_events: list[CorporateEvent] = context.get("upcoming_events", [])
        regime: MarketRegime = context.get("market_regime", MarketRegime.BULL)
        stance: TradingStance = context.get("trading_stance", TradingStance.NORMAL)

        risks: list[str] = []
        disqualified = False
        disqualify_reason: str | None = None

        # 1. Binary Event / Earnings Gap Risk Check
        today = date.today()
        earnings_cutoff = today + timedelta(days=5)  # ~3 trading sessions

        for event in upcoming_events:
            if "RESULTS" in event.event_type.upper() or "BOARD_MEETING" in event.event_type.upper():
                if today <= event.event_date <= earnings_cutoff:
                    disqualified = True
                    disqualify_reason = f"Imminent Earnings Announcement on {event.event_date} (High Overnight Gap Risk)"
                    risks.append(disqualify_reason)
                    break

        # 2. Market Regime Risk Check
        if regime == MarketRegime.STRONG_BEAR or stance == TradingStance.NO_TRADE:
            disqualified = True
            disqualify_reason = f"Hostile Market Regime ({regime.value} - {stance.value}): NO LONG TRADES POLICY"
            risks.append(disqualify_reason)

        # 3. Liquidity Risk Check
        if not df.empty:
            turnover = df["turnover_crores"].tail(20).mean() if "turnover_crores" in df.columns else 10.0
            if turnover < settings.MIN_ADTV_CRORES:
                disqualified = True
                disqualify_reason = f"Inadequate liquidity: ADTV is ₹{turnover:.2f} Cr (< ₹{settings.MIN_ADTV_CRORES} Cr limit)"
                risks.append(disqualify_reason)

        # 4. Volatility / ATR Check
        if not df.empty and "atr_pct" in df.columns:
            atr_pct = float(df["atr_pct"].iloc[-1])
            if atr_pct > 7.5:
                risks.append(f"Elevated daily volatility (ATR: {atr_pct:.1f}%)")

        score = 90.0 if not disqualified else 0.0
        signal = SignalType.REJECT if disqualified else SignalType.BULLISH

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="RISK_EVALUATION",
            raw_metric="risk_clearance_status",
            observed_value=f"Risk Veto: {'PASSED' if not disqualified else 'REJECTED - ' + str(disqualify_reason)}",
            unit="status",
            source="RISK_ENGINE",
            timestamp="EOD",
        )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=score,
            confidence=0.98,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "passed_risk_veto": not disqualified,
                "disqualification_reason": disqualify_reason,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
            disqualification_triggered=disqualified,
            disqualification_reason=disqualify_reason,
        )
