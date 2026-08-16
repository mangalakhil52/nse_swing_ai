"""
Trade Construction Specialist Agent Module.
Calculates actionable trade levels: Entry Trigger Price, Structural Stop Loss, Targets (T1, T2, T3), and Position Size.
Enforces realistic risk-to-reward ratios and strict structural invalidation rules.
"""

from typing import Any
import pandas as pd

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    SymbolMetadata,
    TradeLevels,
)
from src.core.types import AgentStatus, DataFreshness, SignalType
from src.risk.sizing import PositionSizingEngine


class TradeConstructionAgent(BaseAgent):
    """Specialist agent engineering actionable entry, stop loss, targets, and share allocation."""

    def __init__(self):
        super().__init__(agent_name="trade_construction_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        regime_mult: float = context.get("regime_risk_multiplier", 1.0)

        if df.empty or len(df) < 20:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.NEUTRAL,
            )

        cmp = float(df["close"].iloc[-1])
        high_20 = float(df["high"].tail(20).max())
        low_10 = float(df["low"].tail(10).min())
        ema_20 = float(df["ema_20"].iloc[-1]) if "ema_20" in df.columns else cmp * 0.96
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else cmp * 0.025

        # 1. Entry Trigger (Breakout above recent 20-day high with 0.1% buffer or CMP)
        entry_price = round(max(cmp, high_20 * 1.001), 2)

        # 2. Structural Stop Loss (Floor anchored below recent swing low or 20 EMA minus 0.5 ATR buffer)
        structural_floor = min(low_10, ema_20) - (0.5 * atr)
        # Enforce max stop loss cap of settings.MAX_STOP_LOSS_PCT (8%)
        max_sl_floor = entry_price * (1.0 - (settings.MAX_STOP_LOSS_PCT / 100.0))
        stop_loss = round(max(structural_floor, max_sl_floor), 2)

        risk_rupees = round(entry_price - stop_loss, 2)
        risk_pct = round((risk_rupees / entry_price) * 100.0, 2)

        # 3. Targets Calculation (Asymmetric Risk Multiples)
        target_1 = round(entry_price + (risk_rupees * 1.8), 2)
        target_2 = round(entry_price + (risk_rupees * 2.8), 2)
        target_3 = round(entry_price + (risk_rupees * 4.5), 2)

        rr_t1 = round((target_1 - entry_price) / risk_rupees, 2)
        rr_t2 = round((target_2 - entry_price) / risk_rupees, 2)
        rr_t3 = round((target_3 - entry_price) / risk_rupees, 2)

        # 4. Position Sizing
        sizing = PositionSizingEngine.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            regime_risk_multiplier=regime_mult,
        )

        invalidation = f"Daily close below structural stop loss at ₹{stop_loss:.2f} or breach of 20 EMA."

        levels = TradeLevels(
            symbol=symbol,
            current_market_price=cmp,
            entry_trigger_price=entry_price,
            stop_loss_price=stop_loss,
            risk_rupees=risk_rupees,
            risk_percentage=risk_pct,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            risk_reward_t1=rr_t1,
            risk_reward_t2=rr_t2,
            risk_reward_t3=rr_t3,
            position_size_shares=sizing.shares,
            allocated_capital_rupees=sizing.total_capital_allocated,
            invalidation_criteria=invalidation,
        )

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="TRADE_GEOMETRY",
            raw_metric="entry_sl_target_levels",
            observed_value=f"Entry: ₹{entry_price:.2f} | SL: ₹{stop_loss:.2f} ({risk_pct:.1f}%) | T1: ₹{target_1:.2f} (R:R {rr_t1:.1f}) | T2: ₹{target_2:.2f} (R:R {rr_t2:.1f}) | Sizing: {sizing.shares} shares (₹{sizing.total_capital_allocated:,.0f})",
            unit="trade_levels",
            source="TRADE_CONSTRUCTION_ENGINE",
            timestamp="EOD",
        )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=SignalType.BULLISH,
            score=88.0,
            confidence=0.95,
            data_freshness=DataFreshness.RECENT,
            metrics={"trade_levels": levels.model_dump()},
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=[],
        )
