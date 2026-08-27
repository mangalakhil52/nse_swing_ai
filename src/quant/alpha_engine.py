"""Cross-sectional alpha pipeline with regime-aware feature selection."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.types import MarketRegime
from src.quant.advanced_alpha import compute_alpha_features, cross_sectional_zscores


@dataclass(frozen=True)
class AlphaSnapshot:
    symbol: str
    alpha_score: float
    features: dict[str, float]
    regime: MarketRegime


class CrossSectionalAlphaEngine:
    """Ranks candidates using orthogonal, volatility-normalized alpha factors."""

    @staticmethod
    def _regime_weights(regime: MarketRegime) -> dict[str, float]:
        # Weights express relative feature emphasis, not arbitrary probability.
        if regime in {MarketRegime.BULL, MarketRegime.EARLY_BULL}:
            return {"m21": .15, "m63": .30, "m126": .30, "rs": .15, "volume": .10}
        if regime in {MarketRegime.BEAR, MarketRegime.LATE_BEAR}:
            return {"m21": .10, "m63": .15, "m126": .20, "rs": .40, "volume": .15}
        return {"m21": .15, "m63": .25, "m126": .25, "rs": .25, "volume": .10}

    @classmethod
    def rank(
        cls,
        stock_dfs: dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        regime: MarketRegime,
    ) -> list[AlphaSnapshot]:
        features: dict[str, object] = {}
        for symbol, frame in stock_dfs.items():
            try:
                features[symbol] = compute_alpha_features(frame, benchmark_df)
            except (ValueError, KeyError, TypeError):
                continue

        if not features:
            return []

        raw_maps = {
            "m21": {s: f.momentum_21 for s, f in features.items()},
            "m63": {s: f.momentum_63 for s, f in features.items()},
            "m126": {s: f.momentum_126 for s, f in features.items()},
            "rs": {s: f.relative_strength for s, f in features.items()},
            "volume": {s: f.volume_surprise for s, f in features.items()},
        }
        z = {name: cross_sectional_zscores(values) for name, values in raw_maps.items()}
        weights = cls._regime_weights(regime)

        snapshots: list[AlphaSnapshot] = []
        for symbol, f in features.items():
            score = sum(weights[name] * z[name].get(symbol, 0.0) for name in weights)
            # Preserve the richer feature state for downstream probability/risk agents.
            snapshots.append(AlphaSnapshot(
                symbol=symbol,
                alpha_score=round(float(score), 6),
                features={
                    "momentum_21": f.momentum_21,
                    "momentum_63": f.momentum_63,
                    "momentum_126": f.momentum_126,
                    "trend_quality": f.trend_quality,
                    "volatility": f.volatility,
                    "downside_volatility": f.downside_volatility,
                    "volume_surprise": f.volume_surprise,
                    "range_compression": f.range_compression,
                    "breakout_distance": f.breakout_distance,
                    "relative_strength": f.relative_strength,
                },
                regime=regime,
            ))
        return sorted(snapshots, key=lambda x: x.alpha_score, reverse=True)
