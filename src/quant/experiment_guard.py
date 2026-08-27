"""Research promotion guard: prevents unvalidated alpha from reaching live trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: list[str]


def evaluate_promotion(
    out_of_sample_sharpe: float,
    out_of_sample_hit_rate: float,
    max_drawdown: float,
    folds: int,
    benchmark_excess_return: float,
    min_folds: int = 4,
) -> PromotionDecision:
    reasons: list[str] = []
    if folds < min_folds:
        reasons.append("INSUFFICIENT_WALK_FORWARD_FOLDS")
    if out_of_sample_sharpe <= 0.50:
        reasons.append("WEAK_OOS_SHARPE")
    if out_of_sample_hit_rate <= 0.50:
        reasons.append("WEAK_OOS_HIT_RATE")
    if max_drawdown <= -0.35:
        reasons.append("EXCESSIVE_OOS_DRAWDOWN")
    if benchmark_excess_return <= 0:
        reasons.append("NO_BENCHMARK_EXCESS_RETURN")
    return PromotionDecision(not reasons, reasons or ["PROMOTION_GATES_PASSED"])
