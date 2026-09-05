"""
Real Trade Analyzer Module — src/shadow/analyzer.py

Computes analytical performance metrics for real/shadow trades loaded from
the SQLite database repository or trade ledger CSV files.

Calculates:
  - Total Trades, Wins, Losses, Win Rate %
  - Gross Profit (₹), Gross Loss (₹), Profit Factor
  - Expectancy (₹ / trade) & Expectancy (R / trade)
  - Total P&L (₹) & Total P&L (%)
  - Max Drawdown (%)
  - Average Hold Duration (sessions)
  - Exit Reason Breakdown (Target 1, Target 2, Target 3, Stop Loss, Time Stop)
"""

from dataclasses import dataclass
import logging
from typing import Any
import pandas as pd
from sqlalchemy.orm import Session

from src.database.schema import ShadowTradeModel

logger = logging.getLogger(__name__)


@dataclass
class TradeMetricsSummary:
    """Comprehensive performance metrics summary for analyzed real trades."""
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    gross_profit_rupees: float
    gross_loss_rupees: float
    profit_factor: float
    expectancy_rupees: float
    total_pnl_rupees: float
    total_pnl_pct: float
    max_drawdown_pct: float
    avg_holding_sessions: float
    exit_reasons: dict[str, int]


class RealTradeAnalyzer:
    """Analytical engine for real and paper/shadow trade performance."""

    @classmethod
    def analyze_trades_from_db(cls, session: Session, symbol: str | None = None) -> TradeMetricsSummary:
        """Loads shadow trade models from database session and computes trade performance statistics."""
        query = session.query(ShadowTradeModel)
        if symbol:
            query = query.filter(ShadowTradeModel.symbol == symbol.upper().strip())

        db_trades = query.all()
        trade_dicts = []
        for t in db_trades:
            trade_dicts.append({
                "symbol": t.symbol,
                "status": t.status,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "target_1": t.target_1,
                "target_2": t.target_2,
                "target_3": t.target_3,
                "position_size_shares": t.position_size_shares or 100,
                "pnl_rupees": t.pnl_rupees or 0.0,
                "pnl_pct": t.pnl_percentage or 0.0,
                "exit_reason": t.exit_reason or "OPEN",
                "holding_sessions": t.holding_sessions or 1,
            })

        return cls.compute_metrics(trade_dicts)

    @classmethod
    def analyze_trades_from_df(cls, df: pd.DataFrame) -> TradeMetricsSummary:
        """Computes trade performance statistics from a trade ledger DataFrame."""
        if df.empty:
            return cls._empty_summary()

        trade_dicts = df.to_dict(orient="records")
        return cls.compute_metrics(trade_dicts)

    @classmethod
    def compute_metrics(cls, trades: list[dict[str, Any]]) -> TradeMetricsSummary:
        """Calculates comprehensive performance metrics across a list of trade dictionaries."""
        if not trades:
            return cls._empty_summary()

        closed_trades = [t for t in trades if t.get("status") not in ("ACTIVE", "OPEN", "PENDING")]
        if not closed_trades:
            # If all trades are still active, analyze active P&L
            closed_trades = trades

        total_count = len(closed_trades)
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        total_pnl_rs = 0.0
        total_pnl_pct_sum = 0.0
        holding_sessions_list = []
        exit_reasons_counter: dict[str, int] = {}

        pnl_series = []

        for t in closed_trades:
            pnl_rs = float(t.get("pnl_rupees") or 0.0)
            pnl_pct = float(t.get("pnl_pct") or 0.0)
            reason = str(t.get("exit_reason") or "UNKNOWN")
            exit_reasons_counter[reason] = exit_reasons_counter.get(reason, 0) + 1

            if pnl_rs > 0 or pnl_pct > 0:
                wins += 1
                gross_profit += abs(pnl_rs)
            elif pnl_rs < 0 or pnl_pct < 0:
                losses += 1
                gross_loss += abs(pnl_rs)

            total_pnl_rs += pnl_rs
            total_pnl_pct_sum += pnl_pct
            holding_sessions_list.append(int(t.get("holding_sessions") or t.get("exit_session") or 1))
            pnl_series.append(pnl_rs)

        win_rate = (wins / total_count) * 100.0 if total_count > 0 else 0.0
        profit_factor = round(gross_profit / max(gross_loss, 1.0), 2)
        expectancy = round(total_pnl_rs / max(total_count, 1), 2)
        avg_hold = round(sum(holding_sessions_list) / max(len(holding_sessions_list), 1), 1)

        # Max Drawdown calculation from cumulative equity curve
        max_dd_pct = cls._calculate_max_drawdown(pnl_series)

        return TradeMetricsSummary(
            total_trades=total_count,
            wins=wins,
            losses=losses,
            win_rate_pct=round(win_rate, 1),
            gross_profit_rupees=round(gross_profit, 2),
            gross_loss_rupees=round(gross_loss, 2),
            profit_factor=profit_factor,
            expectancy_rupees=expectancy,
            total_pnl_rupees=round(total_pnl_rs, 2),
            total_pnl_pct=round(total_pnl_pct_sum, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            avg_holding_sessions=avg_hold,
            exit_reasons=exit_reasons_counter,
        )

    @staticmethod
    def _calculate_max_drawdown(pnl_list: list[float]) -> float:
        """Calculates maximum peak-to-trough drawdown percentage from cumulative P&L series."""
        if not pnl_list:
            return 0.0

        equity = 100000.0  # Base capital ₹100,000
        equity_curve = [equity]
        for pnl in pnl_list:
            equity += pnl
            equity_curve.append(equity)

        peak = equity_curve[0]
        max_dd = 0.0

        for val in equity_curve:
            if val > peak:
                peak = val
            dd = ((peak - val) / peak) * 100.0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @staticmethod
    def _empty_summary() -> TradeMetricsSummary:
        """Returns empty summary for zero trades."""
        return TradeMetricsSummary(
            total_trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            gross_profit_rupees=0.0,
            gross_loss_rupees=0.0,
            profit_factor=0.0,
            expectancy_rupees=0.0,
            total_pnl_rupees=0.0,
            total_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            avg_holding_sessions=0.0,
            exit_reasons={},
        )
