"""
Execution-Quality & Market Impact Slippage Model.
Calculates liquidity friction, bid-ask spread impact, and execution slippage prior to trade level approval.
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionQualityResult:
    expected_slippage_pct: float
    market_impact_rupees: float
    adjusted_entry_trigger: float
    execution_grade: str  # EXCELLENT, ACCEPTABLE, POOR
    is_executable: bool


class ExecutionQualityModel:
    """Models market impact cost and bid-ask spread slippage based on ADTV and ATR volatility."""

    @classmethod
    def evaluate_execution_quality(
        cls,
        current_price: float,
        entry_trigger_price: float,
        adtv_crores: float,
        allocated_capital_rupees: float,
        atr_14: float,
    ) -> ExecutionQualityResult:
        """
        Calculates execution impact cost:
            Slippage % = Base (0.05%) + 0.10% * (Order Capital / ADTV) * (ATR / CMP)
        """
        adtv_rupees = max(1.0, adtv_crores * 1e7)
        liquidity_ratio = allocated_capital_rupees / adtv_rupees
        volatility_ratio = (atr_14 / max(1.0, current_price))

        slippage_pct = round(0.05 + 0.10 * (liquidity_ratio * 100.0) * (volatility_ratio * 10.0), 3)
        slippage_pct = min(1.50, max(0.05, slippage_pct))

        impact_rupees = round(entry_trigger_price * (slippage_pct / 100.0), 2)
        adjusted_entry = round(entry_trigger_price + impact_rupees, 2)

        if slippage_pct <= 0.15:
            grade = "EXCELLENT"
            executable = True
        elif slippage_pct <= 0.50:
            grade = "ACCEPTABLE"
            executable = True
        else:
            grade = "POOR"
            executable = False

        return ExecutionQualityResult(
            expected_slippage_pct=slippage_pct,
            market_impact_rupees=impact_rupees,
            adjusted_entry_trigger=adjusted_entry,
            execution_grade=grade,
            is_executable=executable,
        )
