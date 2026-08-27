"""Risk-budget based position sizing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSize:
    quantity: int
    capital_rupees: float
    risk_rupees: float
    risk_fraction: float
    capped: bool


def size_position(
    capital_rupees: float,
    entry_price: float,
    stop_price: float,
    risk_budget_fraction: float = 0.0075,
    max_position_fraction: float = 0.20,
) -> PositionSize:
    """Size from loss-to-stop budget, then cap gross capital exposure."""
    if capital_rupees <= 0 or entry_price <= 0 or stop_price <= 0 or entry_price <= stop_price:
        raise ValueError("Invalid long-position inputs")
    if not 0 < risk_budget_fraction < 1 or not 0 < max_position_fraction <= 1:
        raise ValueError("Invalid risk-budget configuration")
    risk_rupees = capital_rupees * risk_budget_fraction
    per_share_risk = entry_price - stop_price
    qty_by_risk = int(risk_rupees // per_share_risk)
    max_capital = capital_rupees * max_position_fraction
    qty_by_cap = int(max_capital // entry_price)
    quantity = max(0, min(qty_by_risk, qty_by_cap))
    capital = quantity * entry_price
    actual_risk = quantity * per_share_risk
    return PositionSize(
        quantity=quantity,
        capital_rupees=capital,
        risk_rupees=actual_risk,
        risk_fraction=actual_risk / capital_rupees,
        capped=qty_by_risk > qty_by_cap,
    )
