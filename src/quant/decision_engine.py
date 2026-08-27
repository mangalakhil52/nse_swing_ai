"""Final deterministic decision gate for alpha, evidence and risk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    action: str
    score: float
    reasons: list[str]


def decide(
    alpha_score: float,
    empirical_probability: float | None,
    net_ev_pct: float | None,
    probability_lower_bound: float | None,
    execution_slippage_pct: float,
    portfolio_allowed: bool,
    data_pit_safe: bool,
    min_alpha: float = 1.0,
    min_probability: float = 0.60,
    min_lower_bound: float = 0.50,
    min_ev_pct: float = 1.5,
    max_slippage_pct: float = 0.30,
) -> Decision:
    """Fail closed unless every critical production gate is satisfied."""
    reasons: list[str] = []
    if not data_pit_safe:
        reasons.append("PIT_UNSAFE")
    if not portfolio_allowed:
        reasons.append("PORTFOLIO_RISK_REJECT")
    if empirical_probability is None or probability_lower_bound is None:
        reasons.append("PROBABILITY_UNAVAILABLE")
    else:
        if empirical_probability < min_probability:
            reasons.append("PROBABILITY_TOO_LOW")
        if probability_lower_bound < min_lower_bound:
            reasons.append("CONFIDENCE_BOUND_TOO_LOW")
    if net_ev_pct is None or net_ev_pct < min_ev_pct:
        reasons.append("NET_EV_TOO_LOW")
    if alpha_score < min_alpha:
        reasons.append("ALPHA_TOO_LOW")
    if execution_slippage_pct > max_slippage_pct:
        reasons.append("EXECUTION_COST_TOO_HIGH")

    if reasons:
        return Decision("NO_TRADE", float(alpha_score), reasons)
    return Decision("TRADE", float(alpha_score), ["ALL_GATES_PASSED"])
