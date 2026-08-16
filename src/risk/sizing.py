"""
Position Sizing Engine Module.
Implements the Fixed Fractional Volatility Model for capital allocation.
Calculates share quantities strictly from account risk parameters and stop-loss distance.
"""

import math
from pydantic import BaseModel, Field

from config.settings import settings
from src.core.types import MarketRegime


class SizingResult(BaseModel):
    shares: int
    entry_price: float
    stop_loss_price: float
    risk_per_share: float
    total_capital_allocated: float
    total_capital_at_risk: float
    capital_allocation_pct: float
    risk_pct_of_account: float


class PositionSizingEngine:
    """Calculates volatility-adjusted position size based on structural stop loss."""

    @classmethod
    def calculate_position_size(
        cls,
        entry_price: float,
        stop_loss_price: float,
        account_capital: float | None = None,
        risk_pct: float | None = None,
        regime_risk_multiplier: float = 1.0,
        max_capital_allocation_pct: float = 20.0,
    ) -> SizingResult:
        capital = account_capital or settings.ACCOUNT_CAPITAL
        base_risk_pct = risk_pct or settings.MAX_RISK_PER_TRADE_PCT

        # Effective risk rupees
        effective_risk_pct = (base_risk_pct / 100.0) * regime_risk_multiplier
        max_allowed_risk_rupees = capital * effective_risk_pct

        risk_per_share = max(0.01, entry_price - stop_loss_price)

        # Raw shares by risk equation
        raw_shares = math.floor(max_allowed_risk_rupees / risk_per_share)

        # Cap by maximum single stock allocation limit (20% of account)
        max_allocated_rupees = capital * (max_capital_allocation_pct / 100.0)
        max_shares_by_capital = math.floor(max_allocated_rupees / entry_price)

        final_shares = max(1, min(raw_shares, max_shares_by_capital))

        total_allocated = final_shares * entry_price
        total_risk_rupees = final_shares * risk_per_share

        return SizingResult(
            shares=final_shares,
            entry_price=round(entry_price, 2),
            stop_loss_price=round(stop_loss_price, 2),
            risk_per_share=round(risk_per_share, 2),
            total_capital_allocated=round(total_allocated, 2),
            total_capital_at_risk=round(total_risk_rupees, 2),
            capital_allocation_pct=round((total_allocated / capital) * 100.0, 2),
            risk_pct_of_account=round((total_risk_rupees / capital) * 100.0, 3),
        )
