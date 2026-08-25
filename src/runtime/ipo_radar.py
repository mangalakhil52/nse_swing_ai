"""Recent-IPO opportunity radar.

Recent listings deserve a separate research track because a normal screen that
requires long history can systematically exclude them. This module does NOT
relax the main Candidate Discovery rules. It creates a separate, conservative
radar using only bars available on/before the decision date.

Listing date is inferred from the first observed OHLCV bar in the supplied
lookback window. A symbol is only labelled a recent listing when the first bar
is recent and the history is continuous enough to make the inference useful.
This deliberately avoids claiming an IPO from an arbitrary data gap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math

import pandas as pd


@dataclass
class IPOOpportunity:
    symbol: str
    inferred_listing_date: str
    listing_age_days: int
    bars: int
    last_close: float
    return_since_first_close_pct: float
    drawdown_from_peak_pct: float
    median_turnover_crores: float
    last_5d_return_pct: float
    volume_ratio_5_vs_20: float | None
    consolidation_pct: float | None
    breakout: bool
    score: float
    track: str = "RECENT_IPO"
    reasons: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RecentIPORadar:
    """Conservative scanner for recent listings excluded by long-history rules."""

    def __init__(
        self,
        as_of_date: date,
        max_age_days: int = 180,
        min_bars: int = 10,
        min_turnover_crores: float = 1.0,
        min_price: float = 20.0,
    ):
        self.as_of_date = as_of_date
        self.max_age_days = max_age_days
        self.min_bars = min_bars
        self.min_turnover_crores = min_turnover_crores
        self.min_price = min_price

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
            out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
        else:
            out = out.reset_index(drop=True)
        required = {"close", "volume"}
        if not required.issubset(out.columns):
            return pd.DataFrame()
        for col in ["close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "turnover_crores" not in out.columns:
            out["turnover_crores"] = out["close"] * out["volume"] / 1e7
        else:
            out["turnover_crores"] = pd.to_numeric(out["turnover_crores"], errors="coerce")
        return out.dropna(subset=["close", "volume", "turnover_crores"])

    def evaluate(self, symbol: str, df: pd.DataFrame) -> IPOOpportunity | None:
        data = self._prepare(df)
        if data.empty or len(data) < self.min_bars:
            return None

        first = data.iloc[0]
        last = data.iloc[-1]
        first_ts = pd.Timestamp(first["timestamp"]).date() if "timestamp" in data.columns else None
        if first_ts is None:
            return None
        age = (self.as_of_date - first_ts).days
        if age < 0 or age > self.max_age_days:
            return None

        # A genuine recent listing should have a reasonably continuous early history.
        # This guards against mistaking a data outage/suspension for an IPO.
        if len(data) >= 11 and "timestamp" in data.columns:
            gaps = data["timestamp"].dt.normalize().diff().dt.days.dropna()
            if not gaps.empty and float(gaps.head(min(15, len(gaps))).max()) > 7:
                return None

        last_close = float(last["close"])
        median_turnover = float(data["turnover_crores"].tail(min(20, len(data))).median())
        if last_close < self.min_price or median_turnover < self.min_turnover_crores:
            return None

        first_close = float(first["close"])
        return_since_first = (last_close / first_close - 1.0) * 100.0 if first_close > 0 else 0.0
        peak = float(data["close"].cummax().iloc[-1])
        drawdown = (last_close / peak - 1.0) * 100.0 if peak > 0 else 0.0

        recent = data.tail(min(5, len(data)))
        last_5 = (float(recent["close"].iloc[-1]) / float(recent["close"].iloc[0]) - 1.0) * 100.0

        vol_ratio = None
        if len(data) >= 20:
            v5 = float(data["volume"].tail(5).mean())
            v20 = float(data["volume"].tail(20).mean())
            vol_ratio = v5 / v20 if v20 > 0 else None

        consolidation = None
        breakout = False
        if len(data) >= 15:
            base = data["close"].tail(15)
            consolidation = (float(base.max()) / float(base.min()) - 1.0) * 100.0
            prior = data["close"].iloc[:-1].tail(10)
            breakout = bool(len(prior) >= 5 and last_close > float(prior.max()))

        score = 50.0
        reasons: list[str] = []
        if return_since_first > 10:
            score += min(15.0, return_since_first * 0.5)
            reasons.append("POSITIVE_POST_LISTING_TREND")
        elif return_since_first < -15:
            score -= 12.0
            reasons.append("WEAK_POST_LISTING_TREND")

        if last_5 > 3:
            score += 8.0
            reasons.append("RECENT_MOMENTUM")
        if vol_ratio is not None and vol_ratio >= 1.3:
            score += 8.0
            reasons.append("VOLUME_EXPANSION")
        if consolidation is not None and consolidation <= 12:
            score += 8.0
            reasons.append("TIGHT_BASE")
        if breakout:
            score += 10.0
            reasons.append("BREAKOUT")
        if drawdown < -12:
            score -= min(12.0, abs(drawdown) * 0.3)
            reasons.append("DEEP_DRAW_DOWN")

        score = max(0.0, min(100.0, score))
        return IPOOpportunity(
            symbol=symbol.upper(),
            inferred_listing_date=first_ts.isoformat(),
            listing_age_days=age,
            bars=len(data),
            last_close=round(last_close, 2),
            return_since_first_close_pct=round(return_since_first, 2),
            drawdown_from_peak_pct=round(drawdown, 2),
            median_turnover_crores=round(median_turnover, 2),
            last_5d_return_pct=round(last_5, 2),
            volume_ratio_5_vs_20=round(vol_ratio, 2) if vol_ratio is not None and math.isfinite(vol_ratio) else None,
            consolidation_pct=round(consolidation, 2) if consolidation is not None else None,
            breakout=breakout,
            score=round(score, 2),
            reasons=reasons,
        )

    def scan(self, market_data: dict[str, pd.DataFrame], limit: int = 25) -> list[IPOOpportunity]:
        rows = []
        for symbol, df in market_data.items():
            item = self.evaluate(symbol, df)
            if item is not None:
                rows.append(item)
        rows.sort(key=lambda x: (-x.score, -x.median_turnover_crores, x.symbol))
        return rows[: max(1, limit)]
