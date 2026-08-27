"""Verified global-market context provider.

This module deliberately fails closed. It never fabricates market prices and
never uses Yahoo Finance. Global cues are optional context; the core regime
engine uses authoritative NSE index/VIX inputs elsewhere.
"""
from __future__ import annotations

from typing import Any


class GlobalMarketProvider:
    """Return only explicitly supplied verified global-market observations."""

    async def fetch_global_indices(self, verified_data: dict[str, Any] | None = None) -> dict[str, Any]:
        if not verified_data:
            return {
                "status": "UNAVAILABLE",
                "reason": "NO_VERIFIED_GLOBAL_SOURCE_CONFIGURED",
                "gift_nifty": None,
                "sp500": None,
                "nasdaq": None,
                "dow": None,
                "nikkei": None,
                "hang_seng": None,
                "brent_crude": None,
                "usdinr": None,
            }
        return {"status": "VERIFIED", **verified_data}

    @classmethod
    def fetch_index_levels(cls, verified_levels: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        """Return verified index levels only; no hardcoded prices are permitted."""
        return verified_levels or {}

    @classmethod
    def generate_day_outlook(cls, global_data: dict[str, Any], news_sentiment: str) -> dict[str, Any]:
        """Generate a neutral unavailable outlook when verified inputs are absent."""
        if global_data.get("status") != "VERIFIED":
            return {
                "gap_type": "UNKNOWN",
                "expected_gap": "UNKNOWN — verified global-market data unavailable",
                "movement_analysis": "Global-market context is unavailable; no directional inference is made.",
                "key_strategy": "Use only verified NSE market-regime and stock-level evidence.",
                "gift_nifty_summary": "GIFT Nifty: unavailable",
            }
        gift = global_data.get("gift_nifty") or {}
        change_pts = gift.get("change_pts")
        change_pct = gift.get("change_pct")
        if change_pts is None or change_pct is None:
            return {
                "gap_type": "UNKNOWN",
                "expected_gap": "UNKNOWN — incomplete verified global data",
                "movement_analysis": "Verified global data is incomplete; no directional inference is made.",
                "key_strategy": "Use only verified NSE market-regime and stock-level evidence.",
                "gift_nifty_summary": "GIFT Nifty: incomplete",
            }
        if change_pts >= 30:
            gap_type, expected = "GAP_UP", f"Gap-Up (approximately +{change_pts:.0f} pts)"
        elif change_pts <= -30:
            gap_type, expected = "GAP_DOWN", f"Gap-Down (approximately {change_pts:.0f} pts)"
        else:
            gap_type, expected = "FLAT_OPEN", "Flat Open / small gap"
        return {
            "gap_type": gap_type,
            "expected_gap": expected,
            "movement_analysis": "Verified global-market inputs indicate a directional opening cue; stock-level confirmation remains required.",
            "key_strategy": "Require stock-level setup and risk gates; do not trade on global cues alone.",
            "gift_nifty_summary": f"GIFT Nifty: {gift.get('price', 'unavailable')} ({change_pts:+.1f} pts / {change_pct:+.2f}%)",
        }
