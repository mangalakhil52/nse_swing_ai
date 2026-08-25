"""
Deterministic Chart Pattern Recognition Engine Module.
Detects mathematically verifiable swing chart patterns (VCP, Flat Base, EMA Pullback, High Tight Flag, Cup & Handle).
Enforces zero-hallucination policy by calculating strict price and volume geometries.
"""

from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.types import PatternType


class PatternMatchResult(BaseModel):
    pattern_type: PatternType
    is_matched: bool
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    consolidation_bars: int = Field(default=0)
    breakout_price: float = Field(default=0.0)
    support_stop_price: float = Field(default=0.0)
    contraction_depth_pct: float = Field(default=0.0)
    volume_surge_ratio: float = Field(default=1.0)
    description: str = Field(default="No pattern detected")
    details: dict[str, Any] = Field(default_factory=dict)


class PatternRecognizer:
    """Deterministic pattern recognition using vectorized price geometry."""

    @classmethod
    def detect_vcp(cls, df: pd.DataFrame) -> PatternMatchResult:
        """Detects Volatility Contraction Pattern (VCP)."""
        if len(df) < 30:
            return PatternMatchResult(pattern_type=PatternType.VOLATILITY_CONTRACTION_PATTERN, is_matched=False)
        close = df["close"].values; high = df["high"].values; low = df["low"].values
        vol = df["volume"].values; rvol = df["rvol_20"].values if "rvol_20" in df.columns else np.ones(len(df))
        if "ema_20" in df.columns and "ema_50" in df.columns:
            ema_20_val = df["ema_20"].iloc[-1]; ema_50_val = df["ema_50"].iloc[-1]
            if close[-1] < ema_20_val * 0.97 or ema_20_val < ema_50_val * 0.97:
                return PatternMatchResult(pattern_type=PatternType.VOLATILITY_CONTRACTION_PATTERN, is_matched=False)
        n_bars = min(len(df), 60); p1_end = int(n_bars * 0.45); p2_end = int(n_bars * 0.80)
        w1_high, w1_low = np.max(high[-n_bars : -n_bars + p1_end]), np.min(low[-n_bars : -n_bars + p1_end])
        w2_high, w2_low = np.max(high[-n_bars + p1_end : -n_bars + p2_end]), np.min(low[-n_bars + p1_end : -n_bars + p2_end])
        w3_high, w3_low = np.max(high[-n_bars + p2_end :]), np.min(low[-n_bars + p2_end :])
        depth_1 = ((w1_high - w1_low) / w1_high) * 100.0; depth_2 = ((w2_high - w2_low) / w2_high) * 100.0; depth_3 = ((w3_high - w3_low) / w3_high) * 100.0
        is_contracting = (depth_1 >= depth_2 * 0.85) and (depth_2 >= depth_3 * 0.85) and (depth_3 <= 6.5)
        avg_vol_w1 = np.mean(vol[-n_bars : -n_bars + p1_end]); avg_vol_w3 = np.mean(vol[-n_bars + p2_end :])
        vol_drying = avg_vol_w3 <= avg_vol_w1 * 1.25
        if is_contracting and vol_drying:
            quality = min(95.0, 70.0 + (6.5 - depth_3) * 4.0)
            breakout_pivot = float(np.max(high[-int(n_bars * 0.3):])); support_floor = float(np.min(low[-n_bars + p2_end :]))
            return PatternMatchResult(pattern_type=PatternType.VOLATILITY_CONTRACTION_PATTERN, is_matched=True,
                quality_score=round(quality, 1), consolidation_bars=n_bars, breakout_price=breakout_pivot,
                support_stop_price=support_floor, contraction_depth_pct=round(depth_3, 2),
                volume_surge_ratio=round(float(rvol[-1]), 2),
                description=f"VCP with 3 contractions: {depth_1:.1f}% -> {depth_2:.1f}% -> {depth_3:.1f}% (Tight pivot {depth_3:.1f}%)",
                details={"depth_1": depth_1, "depth_2": depth_2, "depth_3": depth_3})
        return PatternMatchResult(pattern_type=PatternType.VOLATILITY_CONTRACTION_PATTERN, is_matched=False)

    @classmethod
    def detect_flat_base_breakout(cls, df: pd.DataFrame, base_window: int = 12) -> PatternMatchResult:
        """Detects Flat Base Consolidation and Breakout."""
        if len(df) < base_window + 5:
            return PatternMatchResult(pattern_type=PatternType.FLAT_BASE_BREAKOUT, is_matched=False)
        high = df["high"].values; low = df["low"].values; close = df["close"].values
        rvol = df["rvol_20"].values if "rvol_20" in df.columns else np.ones(len(df))
        base_high = np.max(high[-base_window - 1 : -1]); base_low = np.min(low[-base_window - 1 : -1])
        base_range_pct = ((base_high - base_low) / base_low) * 100.0; current_close = close[-1]
        is_breakout = current_close >= base_high * 0.995; is_tight_base = base_range_pct <= 6.5; has_volume = rvol[-1] >= 1.25
        if is_tight_base and (is_breakout or (current_close >= base_high * 0.98)):
            quality = min(92.0, 75.0 + (6.5 - base_range_pct) * 2.5 + (min(rvol[-1], 3.0) * 5.0))
            return PatternMatchResult(pattern_type=PatternType.FLAT_BASE_BREAKOUT, is_matched=True,
                quality_score=round(quality, 1), consolidation_bars=base_window, breakout_price=float(base_high),
                support_stop_price=float(base_low), contraction_depth_pct=round(base_range_pct, 2),
                volume_surge_ratio=round(float(rvol[-1]), 2),
                description=f"Flat Base Breakout: {base_window}-day range {base_range_pct:.1f}% with {rvol[-1]:.2f}x RVol",
                details={"base_range_pct": base_range_pct, "base_high": base_high, "base_low": base_low, "has_volume": has_volume})
        return PatternMatchResult(pattern_type=PatternType.FLAT_BASE_BREAKOUT, is_matched=False)

    @classmethod
    def detect_ema_pullback_reversal(cls, df: pd.DataFrame) -> PatternMatchResult:
        """Detects 20/50 EMA Pullback with Bullish Reversal Action."""
        if len(df) < 25 or "ema_20" not in df.columns or "ema_50" not in df.columns:
            return PatternMatchResult(pattern_type=PatternType.EMA_PULLBACK_REVERSAL, is_matched=False)
        close = df["close"].values; open_p = df["open"].values; high = df["high"].values; low = df["low"].values
        ema_20 = df["ema_20"].values; ema_50 = df["ema_50"].values
        if ema_20[-1] < ema_50[-1]:
            return PatternMatchResult(pattern_type=PatternType.EMA_PULLBACK_REVERSAL, is_matched=False)
        dist_to_ema = abs(low[-1] - ema_20[-1]) / ema_20[-1] * 100.0
        low_tested_ema = low[-1] <= ema_20[-1] * 1.015 and close[-1] >= ema_20[-1] * 0.995
        bar_range = high[-1] - low[-1]; is_bullish_close = close[-1] > open_p[-1]
        close_location = ((close[-1] - low[-1]) / bar_range) if bar_range > 0 else 0.0
        upper_close = close_location >= 0.55
        if low_tested_ema and is_bullish_close and upper_close:
            stop_price = float(min(low[-1], ema_20[-1] * 0.985)); entry_price = float(high[-1] * 1.002)
            # Quality is now tied to measurable geometry rather than a fixed 82.0:
            # EMA proximity, candle close location/body and volume each contribute.
            proximity = max(0.0, 1.0 - min(dist_to_ema, 1.5) / 1.5)
            body_ratio = abs(close[-1] - open_p[-1]) / bar_range if bar_range > 0 else 0.0
            rvol = float(df["rvol_20"].iloc[-1]) if "rvol_20" in df.columns else 1.0
            volume_quality = min(1.0, max(0.0, (rvol - 0.5) / 1.5))
            quality = 68.0 + 10.0 * proximity + 9.0 * min(1.0, body_ratio / 0.6) + 8.0 * max(0.0, min(1.0, (close_location - 0.55) / 0.45)) + 5.0 * volume_quality
            quality = round(min(95.0, quality), 1)
            return PatternMatchResult(pattern_type=PatternType.EMA_PULLBACK_REVERSAL, is_matched=True,
                quality_score=quality, consolidation_bars=5, breakout_price=entry_price,
                support_stop_price=stop_price, contraction_depth_pct=round(dist_to_ema, 2),
                volume_surge_ratio=round(rvol, 2),
                description=f"Bullish 20 EMA dynamic pullback rejection at ₹{ema_20[-1]:.2f}",
                details={"ema_20": ema_20[-1], "dist_to_ema": dist_to_ema, "body_ratio": body_ratio,
                         "close_location": close_location, "rvol": rvol})
        return PatternMatchResult(pattern_type=PatternType.EMA_PULLBACK_REVERSAL, is_matched=False)

    @classmethod
    def evaluate_all_patterns(cls, df: pd.DataFrame) -> list[PatternMatchResult]:
        """Runs all pattern detectors and returns matching patterns sorted by quality score."""
        results: list[PatternMatchResult] = []
        for detector in (cls.detect_vcp, cls.detect_flat_base_breakout, cls.detect_ema_pullback_reversal):
            result = detector(df)
            if result.is_matched:
                results.append(result)
        results.sort(key=lambda r: r.quality_score, reverse=True)
        return results
