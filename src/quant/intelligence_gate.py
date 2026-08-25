"""Cheap pre-CIO intelligence gate for the dynamic NSE scanner."""
from __future__ import annotations

from dataclasses import dataclass

from src.runtime.ipo_radar import IPOOpportunity
from src.runtime.nse_swing_scan import TechnicalShortlistRow
from src.quant.screener import ScreenerCandidate


@dataclass(frozen=True)
class IntelligenceGateConfig:
    normal_min_technical_score: float = 75.0
    normal_min_trend_score: float = 65.0
    normal_min_rs: float = 0.0
    normal_max_candidates: int = 20
    recent_min_score: float = 75.0
    recent_max_candidates: int = 10


def select_normal(rows: list[TechnicalShortlistRow], candidates: dict[str, ScreenerCandidate], config: IntelligenceGateConfig | None = None) -> list[ScreenerCandidate]:
    """Route technically confirmed, benchmark-supported candidates to CIO."""
    cfg = config or IntelligenceGateConfig()
    eligible: list[tuple[TechnicalShortlistRow, ScreenerCandidate]] = []
    for row in rows:
        candidate = candidates.get(row.symbol)
        if candidate is None or row.signal != "BULLISH" or row.technical_score < cfg.normal_min_technical_score:
            continue
        if candidate.trend_score < cfg.normal_min_trend_score or candidate.mansfield_rs < cfg.normal_min_rs or not row.pit_safe:
            continue
        eligible.append((row, candidate))
    eligible.sort(key=lambda pair: (-pair[0].technical_score, -pair[1].mansfield_rs, pair[0].symbol))
    return [candidate for _, candidate in eligible[:cfg.normal_max_candidates]]


def select_recent(rows: list[IPOOpportunity], candidates: dict[str, ScreenerCandidate], config: IntelligenceGateConfig | None = None) -> list[ScreenerCandidate]:
    """Route high-quality recent listings when the normal long-history gate would miss them."""
    cfg = config or IntelligenceGateConfig()
    eligible: list[tuple[IPOOpportunity, ScreenerCandidate]] = []
    for row in rows:
        candidate = candidates.get(row.symbol)
        if candidate is None or row.score < cfg.recent_min_score:
            continue
        eligible.append((row, candidate))
    eligible.sort(key=lambda pair: (-pair[0].score, -pair[0].median_turnover_crores, pair[0].symbol))
    return [candidate for _, candidate in eligible[:cfg.recent_max_candidates]]
