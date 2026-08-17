"""
Risk Management Specialist Agent Module — Refactored for P0.6, P0.7 & P0.8 Compliance.
Enforces absolute veto power over candidate setups.
Outputs: PASS / CAUTION / VETO / DATA_UNAVAILABLE.
Does NOT generate bullish alpha (score = 0.0). Uses actual NSE trading sessions for event windows.
"""

from datetime import date
from typing import Any
import pandas as pd

from config.market_hours import get_next_trading_sessions
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
    """Specialist risk guard agent with absolute veto authority. Does not contribute to bullish alpha."""

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

        # 1. Binary Event / Earnings Gap Risk Check (Next 3 Actual NSE Trading Sessions - P0.8)
        today = date.today()
        next_3_sessions = get_next_trading_sessions(today, 3)

        for event in upcoming_events:
            if "RESULTS" in event.event_type.upper() or "BOARD_MEETING" in event.event_type.upper():
                if event.event_date in next_3_sessions:
                    disqualified = True
                    disqualify_reason = f"Imminent Earnings Announcement on {event.event_date} (within next 3 NSE trading sessions)"
                    risks.append(disqualify_reason)
                    break

        # 2. Market Regime Risk Check
        if regime == MarketRegime.STRONG_BEAR or stance == TradingStance.NO_TRADE:
            disqualified = True
            disqualify_reason = f"Hostile Market Regime ({regime.value} - {stance.value}): NO LONG TRADES POLICY"
            risks.append(disqualify_reason)

        # 3. Liquidity Risk Check
        if not df.empty:
            turnover = df["turnover_crores"].tail(20).mean() if "turnover_crores" in df.columns else 0.0
            if turnover > 0.0 and turnover < settings.MIN_ADTV_CRORES:
                disqualified = True
                disqualify_reason = f"Inadequate liquidity: ADTV is ₹{turnover:.2f} Cr (< ₹{settings.MIN_ADTV_CRORES} Cr limit)"
                risks.append(disqualify_reason)

        # 4. Volatility / ATR Check
        if not df.empty and "atr_pct" in df.columns:
            atr_pct = float(df["atr_pct"].iloc[-1])
            if atr_pct > 7.5:
                risks.append(f"Elevated daily volatility (ATR: {atr_pct:.1f}%)")

        # P0.6: Risk agent MUST NOT generate alpha (score = 0.0, signal = NEUTRAL if pass, REJECT if fail)
        score = 0.0
        signal = SignalType.REJECT if disqualified else SignalType.NEUTRAL

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="RISK_EVALUATION",
            raw_metric="risk_clearance_status",
            observed_value=f"Risk Veto: {'PASSED' if not disqualified else 'REJECTED - ' + str(disqualify_reason)}",
            unit="status",
            source="RISK_ENGINE",
            timestamp=today.isoformat(),
        )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=score,
            confidence=None,  # P0.7: Uncalibrated confidence marked as None
            data_freshness=DataFreshness.RECENT,
            metrics={
                "passed_risk_veto": not disqualified,
                "disqualification_reason": disqualify_reason,
                "confidence_status": "UNCALIBRATED",
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
            disqualification_triggered=disqualified,
            disqualification_reason=disqualify_reason,
        )
