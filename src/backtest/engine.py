"""
Backtest Engine Module.
Event-driven historical simulation with realistic Indian friction costs,
gap-through-stop handling, partial target exits, and walk-forward validation.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.backtest.friction import IndianFrictionModel

logger = logging.getLogger(__name__)


class BacktestTrade(BaseModel):
    symbol: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    shares: int
    exit_date: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float | None = None
    pnl_rupees: float | None = None
    gross_pnl_rupees: float | None = None
    transaction_cost_rupees: float | None = None
    holding_sessions: int = 0
    max_adverse_excursion_pct: float | None = None
    max_favorable_excursion_pct: float | None = None


class BacktestResult(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_gain_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_rupees: float
    total_pnl_rupees: float
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float | None = None
    trade_details: list[BacktestTrade] = Field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtest engine with Indian market friction and gap handling.
    """

    MAX_HOLDING_SESSIONS = 15
    T1_EXIT_PCT = 0.50  # Exit 50% at Target 1
    T2_EXIT_PCT = 0.30  # Exit 30% at Target 2
    T3_EXIT_PCT = 0.20  # Trail remaining 20% to Target 3

    @classmethod
    def simulate_trade(
        cls,
        symbol: str,
        df: pd.DataFrame,
        entry_date_idx: int,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        target_3: float,
        shares: int,
    ) -> BacktestTrade:
        """
        Simulates one trade's evolution through historical OHLCV bars.
        Handles gap-through-stop, partial target exits, and time stops.
        """
        trade = BacktestTrade(
            symbol=symbol,
            entry_date=str(df.index[entry_date_idx]) if hasattr(df.index, '__iter__') else str(entry_date_idx),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            shares=shares,
        )

        t1_hit = False
        t2_hit = False
        remaining_shares = shares
        realized_pnl = 0.0
        mae_pct = 0.0
        mfe_pct = 0.0
        exit_price = entry_price
        exit_reason = "TIME_STOP"
        holding = 0

        for i in range(entry_date_idx + 1, min(entry_date_idx + cls.MAX_HOLDING_SESSIONS + 1, len(df))):
            row = df.iloc[i]
            high = row["high"]
            low = row["low"]
            close = row["close"]
            holding += 1

            # MAE / MFE tracking
            mae_pct = min(mae_pct, ((low - entry_price) / entry_price) * 100.0)
            mfe_pct = max(mfe_pct, ((high - entry_price) / entry_price) * 100.0)

            # Gap-through-stop check (if open gaps below stop)
            open_p = row.get("open", close)
            if open_p < stop_loss:
                # Gapped through stop — exit at open (realistic slippage)
                exit_price = open_p
                exit_reason = "STOP_LOSS_GAP"
                break

            # Intrabar stop hit
            if low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP_LOSS_HIT"
                break

            # Target 1 partial exit
            if not t1_hit and high >= target_1:
                t1_shares = int(remaining_shares * cls.T1_EXIT_PCT)
                realized_pnl += t1_shares * (target_1 - entry_price)
                remaining_shares -= t1_shares
                t1_hit = True

            # Target 2 partial exit
            if t1_hit and not t2_hit and high >= target_2:
                t2_shares = int(remaining_shares * (cls.T2_EXIT_PCT / (1 - cls.T1_EXIT_PCT)))
                realized_pnl += t2_shares * (target_2 - entry_price)
                remaining_shares -= t2_shares
                t2_hit = True

            # Final target exit
            if t2_hit and high >= target_3:
                realized_pnl += remaining_shares * (target_3 - entry_price)
                remaining_shares = 0
                exit_price = target_3
                exit_reason = "TARGET_3_HIT"
                break

        # Time stop: exit remaining at last close
        if remaining_shares > 0:
            last_close = df.iloc[min(entry_date_idx + holding, len(df) - 1)]["close"]
            realized_pnl += remaining_shares * (last_close - entry_price)
            exit_price = last_close

        # Transaction costs
        costs = IndianFrictionModel.calculate_round_trip(entry_price, exit_price, shares)
        gross_pnl = realized_pnl
        net_pnl = gross_pnl - costs.total_cost_rupees
        pnl_pct = (net_pnl / (entry_price * shares)) * 100.0

        trade.exit_price = round(exit_price, 2)
        trade.exit_reason = exit_reason
        trade.pnl_pct = round(pnl_pct, 2)
        trade.pnl_rupees = round(net_pnl, 2)
        trade.gross_pnl_rupees = round(gross_pnl, 2)
        trade.transaction_cost_rupees = round(costs.total_cost_rupees, 2)
        trade.holding_sessions = holding
        trade.max_adverse_excursion_pct = round(mae_pct, 2)
        trade.max_favorable_excursion_pct = round(mfe_pct, 2)
        trade.exit_date = str(entry_date_idx + holding)

        return trade

    @classmethod
    def run_strategy_backtest(
        cls,
        signal_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        symbol: str = "BACKTEST",
    ) -> BacktestResult:
        """
        Runs batch backtest over all identified entry signals with position sizing.
        signal_df must contain columns: ['entry_idx', 'entry_price', 'stop_loss', 'target_1', 'target_2', 'target_3', 'shares']
        """
        trades: list[BacktestTrade] = []

        for _, sig in signal_df.iterrows():
            entry_idx = int(sig["entry_idx"])
            if entry_idx >= len(ohlcv_df) - 2:
                continue

            trade = cls.simulate_trade(
                symbol=symbol,
                df=ohlcv_df,
                entry_date_idx=entry_idx,
                entry_price=float(sig["entry_price"]),
                stop_loss=float(sig["stop_loss"]),
                target_1=float(sig["target_1"]),
                target_2=float(sig["target_2"]),
                target_3=float(sig["target_3"]),
                shares=int(sig["shares"]),
            )
            trades.append(trade)

        return cls._compute_stats(trades)

    @classmethod
    def _compute_stats(cls, trades: list[BacktestTrade]) -> BacktestResult:
        """Aggregates trade-level stats into strategy performance metrics."""
        if not trades:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate_pct=0.0, avg_gain_pct=0.0, avg_loss_pct=0.0,
                profit_factor=0.0, expectancy_rupees=0.0, total_pnl_rupees=0.0,
                total_pnl_pct=0.0, max_drawdown_pct=0.0,
            )

        pnls = [t.pnl_rupees or 0.0 for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        total_gain = sum(winners)
        total_loss = abs(sum(losers))

        win_rate = (len(winners) / len(trades)) * 100.0 if trades else 0.0
        avg_gain = np.mean([t.pnl_pct or 0.0 for t in trades if (t.pnl_pct or 0) > 0]) if winners else 0.0
        avg_loss = np.mean([t.pnl_pct or 0.0 for t in trades if (t.pnl_pct or 0) <= 0]) if losers else 0.0
        profit_factor = total_gain / total_loss if total_loss > 0 else float("inf")
        expectancy = np.mean(pnls)
        total_pnl = sum(pnls)

        # Drawdown calculation using portfolio equity curve starting at initial capital (P25 & P26)
        initial_capital = 1000000.0
        equity = initial_capital + np.cumsum([0.0] + pnls)
        running_max = np.maximum.accumulate(equity)
        drawdowns = ((equity - running_max) / running_max) * 100.0
        max_drawdown = float(np.min(drawdowns))

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate_pct=round(win_rate, 1),
            avg_gain_pct=round(float(avg_gain), 2),
            avg_loss_pct=round(float(avg_loss), 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 99.0,
            expectancy_rupees=round(float(expectancy), 2),
            total_pnl_rupees=round(total_pnl, 2),
            total_pnl_pct=round((total_pnl / 1000000.0) * 100.0, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            trade_details=trades,
        )
