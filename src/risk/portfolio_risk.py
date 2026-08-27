"""Portfolio-level risk controls for correlated swing signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioRiskDecision:
    allowed: bool
    portfolio_volatility: float
    max_pairwise_correlation: float
    sector_concentration: dict[str, float]
    rejection_reasons: list[str]


def evaluate_portfolio(
    returns_by_symbol: dict[str, pd.Series],
    sectors: dict[str, str],
    proposed_symbols: list[str],
    max_pairwise_correlation: float = 0.85,
    max_sector_weight: float = 0.50,
) -> PortfolioRiskDecision:
    if not proposed_symbols:
        return PortfolioRiskDecision(True, 0.0, 0.0, {}, [])
    columns = {}
    for symbol in proposed_symbols:
        series = returns_by_symbol.get(symbol)
        if series is not None:
            columns[symbol] = pd.to_numeric(series, errors="coerce")
    if not columns:
        return PortfolioRiskDecision(False, 0.0, 0.0, {}, ["NO_RETURN_DATA"])

    frame = pd.DataFrame(columns).dropna(how="all")
    corr = frame.corr().fillna(0.0)
    max_corr = 0.0
    if len(corr) > 1:
        arr = corr.to_numpy()
        max_corr = float(np.max(np.abs(arr - np.eye(len(arr)))))

    equal_weight = np.full(len(columns), 1.0 / len(columns))
    covariance = frame.cov().fillna(0.0).to_numpy() * 252.0
    portfolio_vol = float(np.sqrt(max(equal_weight @ covariance @ equal_weight, 0.0)))

    sector_counts: dict[str, float] = {}
    for symbol in columns:
        sector = sectors.get(symbol, "UNKNOWN")
        sector_counts[sector] = sector_counts.get(sector, 0.0) + 1.0 / len(columns)

    reasons: list[str] = []
    if max_corr > max_pairwise_correlation:
        reasons.append(f"CORRELATION_CLUSTER_TOO_HIGH:{max_corr:.2f}")
    if any(weight > max_sector_weight for weight in sector_counts.values()):
        reasons.append("SECTOR_CONCENTRATION_TOO_HIGH")

    return PortfolioRiskDecision(
        allowed=not reasons,
        portfolio_volatility=portfolio_vol,
        max_pairwise_correlation=max_corr,
        sector_concentration={k: round(v, 4) for k, v in sector_counts.items()},
        rejection_reasons=reasons,
    )
