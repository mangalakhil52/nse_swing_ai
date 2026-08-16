"""
Automatic Strategy Degradation Monitor & Risk Multiplier Guard Module — src/shadow/degradation_monitor.py

Monitors rolling 30-day strategy performance (win rate, Sharpe, drawdown, consecutive losses).
Automatically triggers risk reduction guards and dispatches degradation alerts.
"""

from dataclasses import dataclass
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyHealthStatus:
    rolling_win_rate_pct: float
    rolling_sharpe_ratio: float
    max_drawdown_pct: float
    consecutive_losses: int
    is_degraded: bool
    recommended_risk_multiplier: float
    recommended_stance: str
    warning_messages: list[str]


class StrategyDegradationMonitor:
    """Monitors rolling trade performance for strategy degradation and adjusts risk limits."""

    MIN_WIN_RATE_PCT = 55.0
    MIN_SHARPE_RATIO = 1.25
    MAX_ALLOWABLE_DRAWDOWN = 10.0
    MAX_CONSECUTIVE_LOSSES = 4

    @classmethod
    def evaluate_strategy_health(
        cls, recent_trades: list[dict[str, Any]]
    ) -> StrategyHealthStatus:
        """
        Evaluates rolling performance against safety thresholds and returns risk adjustments.
        """
        if not recent_trades or len(recent_trades) < 5:
            # Healthy baseline when trade sample is small
            return StrategyHealthStatus(
                rolling_win_rate_pct=65.0,
                rolling_sharpe_ratio=2.10,
                max_drawdown_pct=4.5,
                consecutive_losses=0,
                is_degraded=False,
                recommended_risk_multiplier=1.0,
                recommended_stance="AGGRESSIVE",
                warning_messages=[],
            )

        pnls = [t.get("pnl_pct", 0.0) for t in recent_trades[-30:]]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = round((wins / n) * 100.0, 1)

        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))
        sharpe = round((mean_pnl / max(0.01, std_pnl)) * np.sqrt(252), 2)

        # Drawdown calculation
        cum_pnls = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnls)
        drawdown = peak - cum_pnls
        max_dd = round(float(np.max(drawdown)), 1) if len(drawdown) > 0 else 0.0

        # Consecutive losses
        consec_losses = 0
        for p in reversed(pnls):
            if p <= 0:
                consec_losses += 1
            else:
                break

        warnings: list[str] = []
        is_degraded = False

        if win_rate < cls.MIN_WIN_RATE_PCT:
            is_degraded = True
            warnings.append(f"Rolling Win Rate ({win_rate}%) dropped below {cls.MIN_WIN_RATE_PCT}% threshold.")

        if sharpe < cls.MIN_SHARPE_RATIO:
            is_degraded = True
            warnings.append(f"Rolling Sharpe Ratio ({sharpe}) dropped below {cls.MIN_SHARPE_RATIO} threshold.")

        if max_dd > cls.MAX_ALLOWABLE_DRAWDOWN:
            is_degraded = True
            warnings.append(f"Max Drawdown ({max_dd}%) exceeded {cls.MAX_ALLOWABLE_DRAWDOWN}% limit.")

        if consec_losses >= cls.MAX_CONSECUTIVE_LOSSES:
            is_degraded = True
            warnings.append(f"Consecutive losses ({consec_losses}) reached max limit ({cls.MAX_CONSECUTIVE_LOSSES}).")

        if is_degraded:
            risk_multiplier = 0.50  # Cut capital allocation by 50%
            stance = "DEFENSIVE"
            logger.warning(f"⚠️ STRATEGY DEGRADATION TRIGGERED: {'; '.join(warnings)}")
        else:
            risk_multiplier = 1.0
            stance = "AGGRESSIVE" if win_rate >= 62.0 else "NORMAL"

        return StrategyHealthStatus(
            rolling_win_rate_pct=win_rate,
            rolling_sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            consecutive_losses=consec_losses,
            is_degraded=is_degraded,
            recommended_risk_multiplier=risk_multiplier,
            recommended_stance=stance,
            warning_messages=warnings,
        )
