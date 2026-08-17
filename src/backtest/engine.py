"""
Backtest Engine Module — Refactored for P0 Correction: Backtest Independent Trade Level Construction & Parity Verification.

Event-driven historical simulation with realistic Indian friction costs,
gap-through-stop handling, partial target exits, worst-case same-candle conflict resolution,
and 100% independent trade construction parity with the live production TradeConstructionEngine.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.agents.trade_construction_agent import TradeConstructionEngine
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
    executed_buy_value: float = 0.0
    executed_sell_value: float = 0.0


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
    Event-driven backtest engine enforcing 100% trade construction parity with production.
    """

    MAX_HOLDING_SESSIONS = 15
    T1_EXIT_PCT = 0.50  # Exit 50% at Target 1
    T2_EXIT_PCT = 0.30  # Exit 30% at Target 2
    T3_EXIT_PCT = 0.20  # Trail remaining 20% to Target 3

    @classmethod
    def backtest_entry_signal(
        cls,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        entry_idx: int,
        supplied_levels: dict[str, Any] | None = None,
    ) -> tuple[BacktestTrade | None, str | None]:
        """
        Canonical backtest entry point.
        Independently constructs trade levels from point-in-time slice (df.iloc[:entry_idx + 1])
        using TradeConstructionEngine.
        If supplied_levels are passed and differ from canonical levels beyond float tolerance (1e-3),
        rejects trade with PARITY_VIOLATION.
        Returns (BacktestTrade, None) on success, or (None, rejection_reason) on failure.
        """
        if ohlcv_df is None or len(ohlcv_df) == 0 or entry_idx >= len(ohlcv_df):
            return None, "Invalid OHLCV DataFrame or entry_idx out of bounds."

        # 1. Point-in-Time slice up to entry_idx ONLY (t <= T)
        sub_df = ohlcv_df.iloc[: entry_idx + 1]

        # 2. Independently obtain production TradeLevels from TradeConstructionEngine
        canonical_levels, err = TradeConstructionEngine.construct_trade_levels(symbol, sub_df)
        if canonical_levels is None:
            return None, f"Trade construction rejected setup: {err}"

        # 3. Parity Validation against externally supplied trade levels (if provided)
        if supplied_levels:
            for field_key, canonical_val in [
                ("entry_price", canonical_levels.entry_trigger_price),
                ("stop_loss", canonical_levels.stop_loss_price),
                ("target_1", canonical_levels.target_1),
                ("target_2", canonical_levels.target_2),
                ("target_3", canonical_levels.target_3),
            ]:
                if field_key in supplied_levels and supplied_levels[field_key] is not None:
                    supplied_val = float(supplied_levels[field_key])
                    if abs(supplied_val - canonical_val) > 1e-3:
                        return None, (
                            f"PARITY_VIOLATION: Externally supplied {field_key} ({supplied_val}) "
                            f"differs from canonical TradeConstructionEngine value ({canonical_val})."
                        )

        # 4. Simulate trade evolution using strictly canonical trade levels
        trade = cls.simulate_trade(
            symbol=symbol,
            df=ohlcv_df,
            entry_date_idx=entry_idx,
            entry_price=canonical_levels.entry_trigger_price,
            stop_loss=canonical_levels.stop_loss_price,
            target_1=canonical_levels.target_1,
            target_2=canonical_levels.target_2,
            target_3=canonical_levels.target_3,
            shares=canonical_levels.position_size_shares,
        )
        return trade, None

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
        Enforces real date timestamps for entry/exit and worst-case same-candle resolution.
        """
        # Format entry_date as a real timestamp string (YYYY-MM-DD)
        if "timestamp" in df.columns:
            entry_date_str = str(df.iloc[entry_date_idx]["timestamp"])
        elif hasattr(df.index, '__iter__') and len(df.index) > entry_date_idx:
            entry_date_str = str(df.index[entry_date_idx])
        else:
            entry_date_str = f"BAR_{entry_date_idx}"
        if " " in entry_date_str:
            entry_date_str = entry_date_str.split(" ")[0]

        trade = BacktestTrade(
            symbol=symbol,
            entry_date=entry_date_str,
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
        exit_idx = entry_date_idx

        for i in range(entry_date_idx + 1, min(entry_date_idx + cls.MAX_HOLDING_SESSIONS + 1, len(df))):
            row = df.iloc[i]
            open_p = float(row.get("open", row["close"]))
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            holding += 1
            exit_idx = i

            # MAE / MFE tracking
            mae_pct = min(mae_pct, ((low - entry_price) / entry_price) * 100.0)
            mfe_pct = max(mfe_pct, ((high - entry_price) / entry_price) * 100.0)

            # 1. Deterministic WORST-CASE rule: Same-candle conflict (both SL and Target touched in same bar)
            if low <= stop_loss and high >= target_1:
                if open_p < stop_loss:
                    exit_price = open_p
                    exit_reason = "STOP_LOSS_GAP"
                else:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS_HIT"
                remaining_shares = 0
                break

            # 2. Gap-through-stop check (if open gaps below stop)
            if open_p < stop_loss:
                exit_price = open_p
                exit_reason = "STOP_LOSS_GAP"
                remaining_shares = 0
                break

            # 3. Intrabar stop loss hit
            if low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP_LOSS_HIT"
                remaining_shares = 0
                break

            # 4. Target 1 partial exit
            if not t1_hit and high >= target_1:
                t1_shares = int(remaining_shares * cls.T1_EXIT_PCT)
                realized_pnl += t1_shares * (target_1 - entry_price)
                remaining_shares -= t1_shares
                t1_hit = True

            # 5. Target 2 partial exit
            if t1_hit and not t2_hit and high >= target_2:
                t2_shares = int(remaining_shares * (cls.T2_EXIT_PCT / (1 - cls.T1_EXIT_PCT)))
                realized_pnl += t2_shares * (target_2 - entry_price)
                remaining_shares -= t2_shares
                t2_hit = True

            # 6. Final target exit
            if t2_hit and high >= target_3:
                realized_pnl += remaining_shares * (target_3 - entry_price)
                remaining_shares = 0
                exit_price = target_3
                exit_reason = "TARGET_3_HIT"
                break

        # Time stop: exit remaining at last close
        if remaining_shares > 0:
            last_close = float(df.iloc[min(entry_date_idx + holding, len(df) - 1)]["close"])
            realized_pnl += remaining_shares * (last_close - entry_price)
            exit_price = last_close

        # Format exit_date as a real timestamp string (YYYY-MM-DD)
        if "timestamp" in df.columns:
            exit_date_str = str(df.iloc[exit_idx]["timestamp"])
        elif hasattr(df.index, '__iter__') and len(df.index) > exit_idx:
            exit_date_str = str(df.index[exit_idx])
        else:
            exit_date_str = f"BAR_{exit_idx}"
        if " " in exit_date_str:
            exit_date_str = exit_date_str.split(" ")[0]

        # Transaction costs
        costs = IndianFrictionModel.calculate_round_trip(entry_price, exit_price, shares)
        gross_pnl = realized_pnl
        net_pnl = gross_pnl - costs.total_cost_rupees
        pnl_pct = (net_pnl / (entry_price * shares)) * 100.0 if (entry_price * shares) > 0 else 0.0

        trade.exit_price = round(exit_price, 2)
        trade.exit_reason = exit_reason
        trade.pnl_pct = round(pnl_pct, 2)
        trade.pnl_rupees = round(net_pnl, 2)
        trade.gross_pnl_rupees = round(gross_pnl, 2)
        trade.transaction_cost_rupees = round(costs.total_cost_rupees, 2)
        trade.holding_sessions = holding
        trade.max_adverse_excursion_pct = round(mae_pct, 2)
        trade.max_favorable_excursion_pct = round(mfe_pct, 2)
        trade.exit_date = exit_date_str

        return trade

    @classmethod
    def run_strategy_backtest(
        cls,
        signal_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
        symbol: str = "BACKTEST",
    ) -> BacktestResult:
        """
        Runs batch backtest over identified entry signals.
        Independently constructs canonical trade levels for each signal and validates parity.
        """
        trades: list[BacktestTrade] = []

        for _, sig in signal_df.iterrows():
            entry_idx = int(sig["entry_idx"])
            if entry_idx >= len(ohlcv_df) - 2:
                continue

            supplied = sig.to_dict()
            trade, err = cls.backtest_entry_signal(
                symbol=symbol,
                ohlcv_df=ohlcv_df,
                entry_idx=entry_idx,
                supplied_levels=supplied,
            )
            if trade is not None:
                trades.append(trade)
            else:
                logger.warning(f"[{symbol}] Backtest trade at bar {entry_idx} rejected: {err}")

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
