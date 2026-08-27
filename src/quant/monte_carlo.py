"""Trade-path Monte Carlo diagnostics for risk validation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    probability_of_ruin: float
    median_terminal_return: float
    worst_5pct_terminal_return: float
    max_drawdown_95pct: float


def simulate_equity_paths(
    trade_returns: list[float],
    simulations: int = 5000,
    trades_per_path: int | None = None,
    seed: int = 42,
) -> MonteCarloResult:
    """Bootstrap historical trade returns to stress path dependency."""
    clean = np.asarray([x for x in trade_returns if np.isfinite(x)], dtype=float)
    if len(clean) < 30:
        raise ValueError("At least 30 historical trade returns are required")
    n = trades_per_path or len(clean)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean, size=(simulations, n), replace=True)
    equity = np.cumprod(1 + sampled, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity / peaks - 1
    terminal = equity[:, -1] - 1
    max_dd = drawdowns.min(axis=1)
    ruin = np.mean(np.min(equity, axis=1) <= 0.5)
    return MonteCarloResult(
        simulations=simulations,
        probability_of_ruin=float(ruin),
        median_terminal_return=float(np.median(terminal)),
        worst_5pct_terminal_return=float(np.quantile(terminal, 0.05)),
        max_drawdown_95pct=float(np.quantile(max_dd, 0.05)),
    )
