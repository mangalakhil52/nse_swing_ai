"""
Risk Veto Engine Module.
Enforces the 10 Hard Disqualifiers. Overrides numerical scores and immediately rejects candidates violating risk parameters.
"""

from typing import Any
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.models import AgentOutput, SymbolMetadata, TradeLevels
from src.core.types import MarketRegime, TradingStance


class VetoDecision(BaseModel):
    passed: bool
    symbol: str
    rejection_reasons: list[str] = Field(default_factory=list)
    hard_disqualifier_triggered: bool = False


class RiskVetoEngine:
    """Evaluates candidates against absolute risk disqualifiers."""

    @classmethod
    def evaluate_candidate(
        cls,
        symbol_meta: SymbolMetadata,
        agent_outputs: dict[str, AgentOutput],
        trade_levels: TradeLevels | None = None,
        market_regime: MarketRegime = MarketRegime.BULL,
        trading_stance: TradingStance = TradingStance.NORMAL,
    ) -> VetoDecision:
        symbol = symbol_meta.symbol
        rejections: list[str] = []

        # 1. Market Regime Disqualifier
        if market_regime == MarketRegime.STRONG_BEAR or trading_stance == TradingStance.NO_TRADE:
            rejections.append(f"Hostile market regime ({market_regime.value}) - Long trades prohibited.")

        # 2. Check Disqualifiers reported by individual specialist agents
        for agent_name, output in agent_outputs.items():
            if output.disqualification_triggered:
                rejections.append(f"{agent_name}: {output.disqualification_reason}")

        # 3. ASM / GSM Surveillance Check
        if symbol_meta.asm_gsm_stage >= 2:
            rejections.append(f"SEBI Surveillance List ASM/GSM Stage {symbol_meta.asm_gsm_stage}")

        # 4. Trade Geometry & Structural Stop Loss Check
        if trade_levels:
            if trade_levels.risk_percentage > settings.MAX_STOP_LOSS_PCT:
                rejections.append(
                    f"Excessive Stop Loss distance ({trade_levels.risk_percentage:.2f}% > {settings.MAX_STOP_LOSS_PCT}% max allowable limit)"
                )
            if trade_levels.risk_reward_t1 < settings.MIN_RR_TARGET_1:
                rejections.append(
                    f"Inadequate R:R to Target 1 ({trade_levels.risk_reward_t1:.2f} < {settings.MIN_RR_TARGET_1} required)"
                )
            if trade_levels.risk_reward_t2 < settings.MIN_RR_TARGET_2:
                rejections.append(
                    f"Inadequate R:R to Target 2 ({trade_levels.risk_reward_t2:.2f} < {settings.MIN_RR_TARGET_2} required)"
                )

        passed = len(rejections) == 0
        return VetoDecision(
            passed=passed,
            symbol=symbol,
            rejection_reasons=rejections,
            hard_disqualifier_triggered=not passed,
        )
