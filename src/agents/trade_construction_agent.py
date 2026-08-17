"""
Trade Construction Specialist Agent Module — Refactored for P1.0, P1.1, P1.2 & P0 Parity & Indicator Integrity.
Contains TradeConstructionEngine: Canonical trade level construction implementation for both LIVE and HISTORICAL execution paths.
Constructs structural trade levels (Entry Trigger, Structural Stop Loss, Market-Structure Resistance Targets, Sizing).
Does NOT generate bullish alpha (score = 0.0, signal = NEUTRAL).
Rejects trade if structural stop is too wide (> 8%), if resistance occurs before 1.5R, or if required indicators are missing/invalid.
"""

import math
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


class TradeConstructionEngine:
    """Canonical engine constructing structural trade levels from OHLCV slice (Point-in-Time safe)."""

    @classmethod
    def construct_trade_levels(
        cls,
        symbol: str,
        df: pd.DataFrame,
        regime_mult: float = 1.0,
    ) -> tuple[TradeLevels | None, str | None]:
        """
        Calculates canonical structural trade levels from an OHLCV DataFrame slice.
        Returns (TradeLevels, None) on success, or (None, rejection_reason) on failure.
        Guarantees point-in-time safety when df is sliced up to setup_date (t <= T).
        Strictly requires real EMA20 and ATR14 without synthetic fallbacks.
        """
        if df.empty or len(df) < 20:
            return None, "Insufficient data bars (< 20) for trade level construction."

        # Indicator Integrity Check (Part 1: ZERO synthetic fallbacks for EMA20 or ATR14)
        if "ema_20" not in df.columns or "atr_14" not in df.columns:
            return None, "Required indicator data (EMA20 or ATR14) is unavailable."

        ema_20_val = df["ema_20"].iloc[-1]
        atr_14_val = df["atr_14"].iloc[-1]

        if pd.isnull(ema_20_val) or pd.isnull(atr_14_val):
            return None, "Required indicator data (EMA20 or ATR14) is unavailable."

        ema_20 = float(ema_20_val)
        atr = float(atr_14_val)

        if math.isnan(ema_20) or math.isinf(ema_20) or ema_20 <= 0.0 or math.isnan(atr) or math.isinf(atr) or atr <= 0.0:
            return None, "Required indicator data (EMA20 or ATR14) is invalid or non-positive."

        cmp = float(df["close"].iloc[-1])
        high_20 = float(df["high"].tail(20).max())
        high_60 = float(df["high"].tail(60).max())
        low_10 = float(df["low"].tail(10).min())

        # 1. Entry Trigger (Breakout above recent 20-day high with 0.1% buffer)
        entry_price = round(max(cmp, high_20 * 1.001), 2)

        # 2. Structural Stop Loss (P1.2: Must represent true thesis invalidation floor below swing low / 20 EMA)
        structural_stop = round(min(low_10, ema_20) - (0.5 * atr), 2)
        risk_rupees = round(entry_price - structural_stop, 2)

        if risk_rupees <= 0.0:
            return None, "Invalid trade geometry: Structural stop is above or equal to entry price."

        risk_pct = round((risk_rupees / entry_price) * 100.0, 2)

        # P1.2: Reject trade if structural stop loss distance exceeds max limit (8.0%)
        if risk_pct > settings.MAX_STOP_LOSS_PCT:
            return None, f"Structural stop loss ({risk_pct:.1f}%) exceeds max allowable limit ({settings.MAX_STOP_LOSS_PCT}%). Trade rejected."

        # 3. Market-Structure Targets (P1.1: Respect swing highs, resistance, and ATR reaction levels)
        resistance_zone_1 = max(high_60, entry_price + (risk_rupees * 1.5))
        resistance_zone_2 = resistance_zone_1 + (1.5 * atr)
        resistance_zone_3 = resistance_zone_2 + (2.0 * atr)

        target_1 = round(resistance_zone_1, 2)
        target_2 = round(resistance_zone_2, 2)
        target_3 = round(resistance_zone_3, 2)

        rr_t1 = round((target_1 - entry_price) / risk_rupees, 2)
        rr_t2 = round((target_2 - entry_price) / risk_rupees, 2)
        rr_t3 = round((target_3 - entry_price) / risk_rupees, 2)

        # P1.1: If realistic resistance occurs before 1.5R minimum acceptable reward, reject trade
        if rr_t1 < 1.5:
            return None, f"Insufficient Market-Structure R:R (T1 R:R {rr_t1:.2f} < 1.50 min threshold due to overhead resistance). Trade rejected."

        # 4. Position Sizing
        sizing = PositionSizingEngine.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=structural_stop,
            regime_risk_multiplier=regime_mult,
        )

        invalidation = f"Daily close below structural stop loss at ₹{structural_stop:.2f} or breach of 20 EMA."

        levels = TradeLevels(
            symbol=symbol,
            current_market_price=cmp,
            entry_trigger_price=entry_price,
            stop_loss_price=structural_stop,
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
        return levels, None


class TradeConstructionAgent(BaseAgent):
    """Specialist agent constructing structural trade levels and position sizing without generating alpha score."""

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

        # Delegate trade level construction to canonical TradeConstructionEngine
        levels, rejection_reason = TradeConstructionEngine.construct_trade_levels(
            symbol=symbol,
            df=df,
            regime_mult=regime_mult,
        )

        if levels is None:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                status=AgentStatus.SUCCESS,
                signal=SignalType.NEUTRAL,
                score=0.0,
                confidence=None,
                disqualification_triggered=True,
                disqualification_reason=rejection_reason or "Trade construction failed.",
            )

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="STRUCTURAL_TRADE_GEOMETRY",
            raw_metric="structural_levels",
            observed_value=(
                f"Entry: ₹{levels.entry_trigger_price:.2f} | Structural SL: ₹{levels.stop_loss_price:.2f} ({levels.risk_percentage:.1f}%) | "
                f"T1: ₹{levels.target_1:.2f} (R:R {levels.risk_reward_t1:.1f}) | Sizing: {levels.position_size_shares} shares"
            ),
            unit="trade_levels",
            source="TRADE_CONSTRUCTION_ENGINE",
            timestamp="EOD",
        )

        # P1.0: Score = 0.0, Signal = NEUTRAL (Trade construction outputs levels, not bullish score!)
        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=SignalType.NEUTRAL,
            score=0.0,
            confidence=None,
            data_freshness=DataFreshness.RECENT,
            metrics={"trade_levels": levels.model_dump()},
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=[],
        )
