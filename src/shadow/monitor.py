"""
Shadow Trade Ledger Monitor Module.
Tracks all paper trades from recommendation date through exit, marks targets hit/stop-loss triggered,
and calculates rolling performance metrics for system validation.
"""

import logging
from datetime import date, timedelta
from typing import Any
import pandas as pd

from src.core.types import ExitReason, TradeStatus

logger = logging.getLogger(__name__)


class ShadowTradeUpdate:
    """Processes intraday/EOD price updates against open paper trades."""

    @staticmethod
    def check_and_update(
        trade: dict[str, Any],
        daily_bar: dict[str, float],
        session_count: int,
    ) -> dict[str, Any]:
        """
        Evaluates a single EOD bar against an open shadow trade position.
        Updates trade status if stop/target hit.

        trade dict must contain: symbol, entry_price, stop_loss, target_1, target_2, target_3, status
        daily_bar must contain: open, high, low, close
        """
        if trade.get("status") != "ACTIVE":
            return trade

        updated = dict(trade)
        high = daily_bar["high"]
        low = daily_bar["low"]
        open_p = daily_bar.get("open", daily_bar["close"])

        # Gap-through-stop (opens below stop)
        if open_p < trade["stop_loss"]:
            updated["exit_price"] = round(open_p, 2)
            updated["exit_reason"] = ExitReason.STOP_LOSS.value
            updated["status"] = TradeStatus.STOPPED_OUT.value
            updated["exit_session"] = session_count
            pnl_pct = ((open_p - trade["entry_price"]) / trade["entry_price"]) * 100.0
            updated["pnl_pct"] = round(pnl_pct, 2)
            logger.info(f"[SHADOW] {trade['symbol']}: Gap-through-stop at ₹{open_p:.2f} (Session {session_count})")
            return updated

        # Intrabar stop hit
        if low <= trade["stop_loss"]:
            updated["exit_price"] = round(trade["stop_loss"], 2)
            updated["exit_reason"] = ExitReason.STOP_LOSS.value
            updated["status"] = TradeStatus.STOPPED_OUT.value
            updated["exit_session"] = session_count
            pnl_pct = ((trade["stop_loss"] - trade["entry_price"]) / trade["entry_price"]) * 100.0
            updated["pnl_pct"] = round(pnl_pct, 2)
            logger.info(f"[SHADOW] {trade['symbol']}: Stop loss hit at ₹{trade['stop_loss']:.2f} (Session {session_count})")
            return updated

        # Target 1 hit
        if not trade.get("t1_hit") and high >= trade["target_1"]:
            updated["t1_hit"] = True
            updated["t1_session"] = session_count
            updated["status"] = TradeStatus.TARGET_1_HIT.value
            logger.info(f"[SHADOW] {trade['symbol']}: Target 1 hit ₹{trade['target_1']:.2f} (Session {session_count})")

        # Target 2 hit
        if trade.get("t1_hit") and not trade.get("t2_hit") and high >= trade["target_2"]:
            updated["t2_hit"] = True
            updated["t2_session"] = session_count
            updated["status"] = TradeStatus.TARGET_2_HIT.value
            logger.info(f"[SHADOW] {trade['symbol']}: Target 2 hit ₹{trade['target_2']:.2f} (Session {session_count})")

        # Final target hit
        if trade.get("t2_hit") and not trade.get("t3_hit") and high >= trade["target_3"]:
            updated["t3_hit"] = True
            updated["exit_price"] = round(trade["target_3"], 2)
            updated["exit_reason"] = ExitReason.TARGET_3.value
            updated["status"] = TradeStatus.TARGET_3_HIT.value
            updated["exit_session"] = session_count
            pnl_pct = ((trade["target_3"] - trade["entry_price"]) / trade["entry_price"]) * 100.0
            updated["pnl_pct"] = round(pnl_pct, 2)
            logger.info(f"[SHADOW] {trade['symbol']}: Full target ₹{trade['target_3']:.2f} HIT (Session {session_count})")

        # Time stop (15 sessions)
        if session_count >= 15 and updated.get("status") == "ACTIVE":
            close = daily_bar["close"]
            updated["exit_price"] = round(close, 2)
            updated["exit_reason"] = ExitReason.TIME_STOP.value
            updated["status"] = TradeStatus.TIME_EXPIRED.value
            updated["exit_session"] = session_count
            pnl_pct = ((close - trade["entry_price"]) / trade["entry_price"]) * 100.0
            updated["pnl_pct"] = round(pnl_pct, 2)
            logger.info(f"[SHADOW] {trade['symbol']}: Time stop expired at ₹{close:.2f} (Session {session_count})")

        return updated


class ShadowPerformanceReport:
    """Calculates rolling P&L and win-rate for closed shadow trades."""

    @staticmethod
    def generate_report(closed_trades: list[dict[str, Any]]) -> dict[str, Any]:
        if not closed_trades:
            return {"total": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0}

        pnls = [t.get("pnl_pct", 0.0) for t in closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        return {
            "total": len(closed_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(closed_trades) * 100.0, 1),
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 2),
            "avg_gain_pct": round(sum(winners) / len(winners), 2) if winners else 0.0,
            "avg_loss_pct": round(sum(losers) / len(losers), 2) if losers else 0.0,
            "total_pnl_pct": round(sum(pnls), 2),
        }
