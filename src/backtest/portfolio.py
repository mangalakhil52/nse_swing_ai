"""
Portfolio Capital & Position Accounting Module — src/backtest/portfolio.py (P0 Fix #11A & #11B)

Enforces strict portfolio-level capital accounting, risk-based position sizing,
portfolio open-risk budgets, chronological event processing, survivorship-safe position tracking,
friction cost deduction, and double-counting prevention.
Composes single-trade BacktestEngine, canonical TradeConstructionEngine, and PositionSizingEngine.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Any
import pandas as pd

from src.agents.trade_construction_agent import TradeConstructionEngine
from src.backtest.engine import BacktestTrade, BacktestResult, BacktestEngine
from src.backtest.friction import IndianFrictionModel
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    invested_value: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    entry_cost: float
    entry_idx: int = 0
    t1_hit: bool = False
    t2_hit: bool = False
    remaining_shares: int = 0
    realized_pnl_so_far: float = 0.0
    holding_sessions: int = 0
    max_high: float = 0.0
    min_low: float = 0.0
    executed_sell_value: float = 0.0

    def __post_init__(self):
        if self.remaining_shares == 0:
            self.remaining_shares = self.shares
        if self.max_high == 0.0:
            self.max_high = self.entry_price
        if self.min_low == 0.0:
            self.min_low = self.entry_price


@dataclass
class DailyPortfolioSnapshot:
    date: str
    cash_available: float
    invested_capital: float
    market_value: float
    total_equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int
    exposure_pct: float


@dataclass
class PortfolioState:
    initial_capital: float = 1000000.0
    cash_available: float = 1000000.0
    invested_capital: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_equity: float = 1000000.0
    max_risk_per_trade_pct: float = 0.50
    max_total_open_risk_pct: float = 2.00
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    completed_trades: list[BacktestTrade] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    equity_curve: list[DailyPortfolioSnapshot] = field(default_factory=list)

    def __post_init__(self):
        if self.cash_available == 1000000.0 and self.initial_capital != 1000000.0:
            self.cash_available = float(self.initial_capital)
        if self.total_equity == 1000000.0 and self.initial_capital != 1000000.0:
            self.total_equity = float(self.initial_capital)

    @property
    def current_total_open_risk(self) -> float:
        """Calculates aggregate stop-loss risk across all active open positions based on remaining shares."""
        return sum(
            (pos.entry_price - pos.stop_loss) * pos.remaining_shares
            for pos in self.open_positions.values()
        )


class PortfolioBacktestEngine:
    """
    Chronological portfolio walk-forward backtest simulator with strict capital accounting and risk limits.
    """

    MAX_HOLDING_SESSIONS = 15
    T1_EXIT_PCT = 0.50  # Exit 50% at Target 1
    T2_EXIT_PCT = 0.30  # Exit 30% at Target 2
    T3_EXIT_PCT = 0.20  # Trail remaining 20% to Target 3

    @classmethod
    def calculate_entry_friction(cls, entry_price: float, shares: int) -> float:
        """Calculates buy-side transaction friction costs (STT buy, SEBI, exchange, stamp duty, slippage)."""
        entry_val = entry_price * shares
        if entry_val <= 0:
            return 0.0
        stt = entry_val * IndianFrictionModel.STT_BUY
        sebi = entry_val * IndianFrictionModel.SEBI_CHARGE
        txn = entry_val * IndianFrictionModel.NSE_TXN_CHARGE
        stamp = entry_val * IndianFrictionModel.STAMP_DUTY
        slip = entry_val * IndianFrictionModel.DEFAULT_SLIPPAGE_PCT
        return round(stt + sebi + txn + stamp + slip, 2)

    @classmethod
    def calculate_exit_friction(cls, exit_price: float, shares: int) -> float:
        """Calculates sell-side transaction friction costs (STT sell, SEBI, exchange, slippage)."""
        exit_val = exit_price * shares
        if exit_val <= 0:
            return 0.0
        stt = exit_val * IndianFrictionModel.STT_SELL
        sebi = exit_val * IndianFrictionModel.SEBI_CHARGE
        txn = exit_val * IndianFrictionModel.NSE_TXN_CHARGE
        slip = exit_val * IndianFrictionModel.DEFAULT_SLIPPAGE_PCT
        return round(stt + sebi + txn + slip, 2)

    @classmethod
    def run_portfolio_backtest(
        cls,
        stock_dfs: dict[str, pd.DataFrame],
        initial_capital: float = 1000000.0,
        max_risk_per_trade_pct: float = 0.50,
        max_total_open_risk_pct: float = 2.00,
    ) -> tuple[PortfolioState, BacktestResult]:
        """
        Runs full chronological portfolio walk-forward backtest across stock_dfs.
        Enforces strict capital limits, risk-based position sizing, aggregate open-risk ceilings,
        single position per symbol, friction deduction, and point-in-time safety.
        """
        portfolio = PortfolioState(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            max_total_open_risk_pct=max_total_open_risk_pct,
        )

        if not stock_dfs:
            stats = BacktestEngine._compute_stats([])
            return portfolio, stats

        # Precompute indicators & map timestamps
        enriched_dfs: dict[str, pd.DataFrame] = {}
        all_timestamps_set = set()

        for sym, df in stock_dfs.items():
            if df is None or len(df) < 50:
                continue
            e_df = TechnicalIndicators.compute_all_indicators(df.copy())
            enriched_dfs[sym] = e_df
            if "timestamp" in e_df.columns:
                all_timestamps_set.update(pd.to_datetime(e_df["timestamp"]))
            else:
                all_timestamps_set.update(e_df.index)

        sorted_timestamps = sorted(list(all_timestamps_set))

        # Main chronological daily loop
        for current_ts in sorted_timestamps:
            date_str = pd.to_datetime(current_ts).strftime("%Y-%m-%d")

            # -------------------------------------------------------------
            # STEP 1: Process Exits for Open Positions
            # -------------------------------------------------------------
            symbols_to_close: list[str] = []

            for sym, pos in list(portfolio.open_positions.items()):
                df_sym = enriched_dfs.get(sym)
                if df_sym is None:
                    continue

                # Locate current session bar
                if "timestamp" in df_sym.columns:
                    mask = pd.to_datetime(df_sym["timestamp"]) == current_ts
                else:
                    mask = df_sym.index == current_ts

                if not mask.any():
                    continue

                row = df_sym[mask].iloc[0]
                high_p = float(row["high"])
                low_p = float(row["low"])
                open_p = float(row.get("open", row["close"]))
                close_p = float(row["close"])

                pos.holding_sessions += 1
                if high_p > pos.max_high:
                    pos.max_high = high_p
                if low_p < pos.min_low:
                    pos.min_low = low_p

                # 1. Same-candle conflict (both SL and Target touched) -> Worst-case SL hit
                if low_p <= pos.stop_loss and high_p >= pos.target_1:
                    exit_price = open_p if open_p < pos.stop_loss else pos.stop_loss
                    exit_reason = "STOP_LOSS_GAP" if open_p < pos.stop_loss else "STOP_LOSS_HIT"
                    cls._close_position(portfolio, pos, exit_price, exit_reason, date_str)
                    symbols_to_close.append(sym)
                    continue

                # 2. Gap down below stop loss
                if open_p < pos.stop_loss:
                    cls._close_position(portfolio, pos, open_p, "STOP_LOSS_GAP", date_str)
                    symbols_to_close.append(sym)
                    continue

                # 3. Intrabar stop loss hit
                if low_p <= pos.stop_loss:
                    cls._close_position(portfolio, pos, pos.stop_loss, "STOP_LOSS_HIT", date_str)
                    symbols_to_close.append(sym)
                    continue

                # 4. Target 1 partial exit
                if not pos.t1_hit and high_p >= pos.target_1:
                    t1_shares = int(pos.remaining_shares * cls.T1_EXIT_PCT)
                    if t1_shares > 0:
                        t1_friction = cls.calculate_exit_friction(pos.target_1, t1_shares)
                        t1_proceeds = (pos.target_1 * t1_shares) - t1_friction
                        portfolio.cash_available += t1_proceeds
                        portfolio.invested_capital -= (pos.entry_price * t1_shares)
                        realized_gain = (pos.target_1 - pos.entry_price) * t1_shares - t1_friction
                        portfolio.realized_pnl += realized_gain
                        pos.realized_pnl_so_far += realized_gain
                        pos.executed_sell_value += (pos.target_1 * t1_shares)
                        pos.remaining_shares -= t1_shares
                        pos.t1_hit = True

                # 5. Target 2 partial exit
                if pos.t1_hit and not pos.t2_hit and high_p >= pos.target_2:
                    t2_shares = int(pos.remaining_shares * (cls.T2_EXIT_PCT / (1 - cls.T1_EXIT_PCT)))
                    if t2_shares > 0:
                        t2_friction = cls.calculate_exit_friction(pos.target_2, t2_shares)
                        t2_proceeds = (pos.target_2 * t2_shares) - t2_friction
                        portfolio.cash_available += t2_proceeds
                        portfolio.invested_capital -= (pos.entry_price * t2_shares)
                        realized_gain = (pos.target_2 - pos.entry_price) * t2_shares - t2_friction
                        portfolio.realized_pnl += realized_gain
                        pos.realized_pnl_so_far += realized_gain
                        pos.executed_sell_value += (pos.target_2 * t2_shares)
                        pos.remaining_shares -= t2_shares
                        pos.t2_hit = True

                # 6. Target 3 final exit
                if pos.t2_hit and high_p >= pos.target_3:
                    cls._close_position(portfolio, pos, pos.target_3, "TARGET_3_HIT", date_str)
                    symbols_to_close.append(sym)
                    continue

                # 7. Time stop (max holding sessions reached)
                if pos.holding_sessions >= cls.MAX_HOLDING_SESSIONS and pos.remaining_shares > 0:
                    cls._close_position(portfolio, pos, close_p, "TIME_STOP", date_str)
                    symbols_to_close.append(sym)
                    continue

            for sym in symbols_to_close:
                if sym in portfolio.open_positions:
                    del portfolio.open_positions[sym]

            # -------------------------------------------------------------
            # STEP 2: Process New Candidate Signals on current_ts
            # -------------------------------------------------------------
            for sym, df_sym in enriched_dfs.items():
                if "timestamp" in df_sym.columns:
                    mask = pd.to_datetime(df_sym["timestamp"]) == current_ts
                else:
                    mask = df_sym.index == current_ts

                if not mask.any():
                    continue

                idx_list = df_sym.index[mask].tolist()
                bar_idx = df_sym.index.get_loc(idx_list[0]) if not isinstance(idx_list[0], int) else idx_list[0]

                if bar_idx < 50:
                    continue

                sub_df = df_sym.iloc[: bar_idx + 1]  # Point-in-time slice (t <= T)
                patterns = PatternRecognizer.evaluate_all_patterns(sub_df)

                for p in patterns:
                    if p.is_matched and p.quality_score >= 75.0:
                        # 1. Position Identity Check: Default 1 position per symbol
                        if sym in portfolio.open_positions:
                            portfolio.rejection_reasons.append(
                                f"POSITION_ALREADY_OPEN: Position already active for {sym} on {date_str}."
                            )
                            break

                        # 2. Canonical Trade Level Construction
                        canonical_levels, err = TradeConstructionEngine.construct_trade_levels(sym, sub_df)
                        if canonical_levels is None:
                            if "stop is above or equal" in (err or "") or "INVALID_RISK_GEOMETRY" in (err or ""):
                                portfolio.rejection_reasons.append(f"INVALID_RISK_GEOMETRY: {err}")
                            else:
                                portfolio.rejection_reasons.append(f"TRADE_REJECTED: {err}")
                            break

                        entry_price = canonical_levels.entry_trigger_price
                        stop_loss = canonical_levels.stop_loss_price

                        # 3. Explicit Risk Geometry Validation
                        if entry_price <= 0.0 or stop_loss <= 0.0 or stop_loss >= entry_price:
                            portfolio.rejection_reasons.append(
                                f"INVALID_RISK_GEOMETRY: Entry {entry_price} <= 0, Stop {stop_loss} <= 0, or Stop >= Entry for {sym} on {date_str}."
                            )
                            break

                        risk_per_share = entry_price - stop_loss
                        if risk_per_share <= 0.0:
                            portfolio.rejection_reasons.append(
                                f"INVALID_RISK_GEOMETRY: Risk per share {risk_per_share:.2f} <= 0 for {sym} on {date_str}."
                            )
                            break

                        # 4. Risk Budget Sizing (uses current total equity as portfolio basis)
                        portfolio_equity = portfolio.total_equity
                        max_trade_risk = portfolio_equity * (portfolio.max_risk_per_trade_pct / 100.0)
                        max_shares_by_risk = math.floor(max_trade_risk / risk_per_share)

                        if max_shares_by_risk < 1:
                            portfolio.rejection_reasons.append(
                                f"RISK_BUDGET_TOO_SMALL: Max shares by risk ({max_shares_by_risk}) < 1 for {sym} on {date_str}."
                            )
                            break

                        # 5. Cash Sizing Limit
                        raw_cash_shares = math.floor(portfolio.cash_available / (entry_price * 1.002))
                        while raw_cash_shares > 0 and ((entry_price * raw_cash_shares) + cls.calculate_entry_friction(entry_price, raw_cash_shares)) > portfolio.cash_available:
                            raw_cash_shares -= 1

                        if raw_cash_shares < 1:
                            portfolio.rejection_reasons.append(
                                f"INSUFFICIENT_PORTFOLIO_CAPITAL: Cash ₹{portfolio.cash_available:,.2f} cannot afford 1 share for {sym} on {date_str}."
                            )
                            break

                        # Final Shares = min(max_shares_by_risk, max_shares_by_cash, canonical_position_size)
                        shares = min(max_shares_by_risk, raw_cash_shares, canonical_levels.position_size_shares)
                        if shares < 1:
                            if max_shares_by_risk < 1:
                                portfolio.rejection_reasons.append(
                                    f"RISK_BUDGET_TOO_SMALL: Final shares ({shares}) < 1 for {sym} on {date_str}."
                                )
                            else:
                                portfolio.rejection_reasons.append(
                                    f"INSUFFICIENT_PORTFOLIO_CAPITAL: Final shares ({shares}) < 1 for {sym} on {date_str}."
                                )
                            break

                        # 6. Aggregate Open-Risk Limit Check
                        current_open_risk = portfolio.current_total_open_risk
                        new_trade_risk = risk_per_share * shares
                        projected_open_risk = current_open_risk + new_trade_risk
                        max_total_open_risk = portfolio_equity * (portfolio.max_total_open_risk_pct / 100.0)

                        if projected_open_risk > max_total_open_risk:
                            portfolio.rejection_reasons.append(
                                f"MAX_PORTFOLIO_RISK_EXCEEDED: Projected open risk ₹{projected_open_risk:,.2f} > Max allowed ₹{max_total_open_risk:,.2f} ({portfolio.max_total_open_risk_pct}%) for {sym} on {date_str}."
                            )
                            break

                        # 7. Accept Entry: Deduct required capital from cash, record open position
                        entry_cost = cls.calculate_entry_friction(entry_price, shares)
                        required_capital = (entry_price * shares) + entry_cost

                        portfolio.cash_available -= required_capital
                        portfolio.invested_capital += (entry_price * shares)

                        new_pos = OpenPosition(
                            symbol=sym,
                            entry_date=date_str,
                            entry_price=entry_price,
                            shares=shares,
                            invested_value=round(entry_price * shares, 2),
                            stop_loss=stop_loss,
                            target_1=canonical_levels.target_1,
                            target_2=canonical_levels.target_2,
                            target_3=canonical_levels.target_3,
                            entry_cost=entry_cost,
                            entry_idx=bar_idx,
                        )
                        portfolio.open_positions[sym] = new_pos
                        break

            # -------------------------------------------------------------
            # STEP 3: Mark Portfolio State & Equity Curve
            # -------------------------------------------------------------
            unrealized_sum = 0.0
            market_val_sum = 0.0

            for sym, pos in portfolio.open_positions.items():
                df_sym = enriched_dfs.get(sym)
                if df_sym is not None:
                    if "timestamp" in df_sym.columns:
                        mask = pd.to_datetime(df_sym["timestamp"]) <= current_ts
                    else:
                        mask = df_sym.index <= current_ts
                    if mask.any():
                        c_price = float(df_sym[mask].iloc[-1]["close"])
                    else:
                        c_price = pos.entry_price
                else:
                    c_price = pos.entry_price

                pos_market_val = c_price * pos.remaining_shares
                pos_unrealized = (c_price - pos.entry_price) * pos.remaining_shares
                market_val_sum += pos_market_val
                unrealized_sum += pos_unrealized

            portfolio.unrealized_pnl = round(unrealized_sum, 2)
            # Core Capital Accounting Identity: Total Equity = Cash Available + Market Value of Open Positions
            portfolio.total_equity = round(portfolio.cash_available + market_val_sum, 2)

            snapshot = DailyPortfolioSnapshot(
                date=date_str,
                cash_available=round(portfolio.cash_available, 2),
                invested_capital=round(portfolio.invested_capital, 2),
                market_value=round(market_val_sum, 2),
                total_equity=round(portfolio.total_equity, 2),
                realized_pnl=round(portfolio.realized_pnl, 2),
                unrealized_pnl=round(portfolio.unrealized_pnl, 2),
                open_positions=len(portfolio.open_positions),
                exposure_pct=round((market_val_sum / portfolio.total_equity) * 100.0, 2) if portfolio.total_equity > 0 else 0.0,
            )
            portfolio.equity_curve.append(snapshot)

        stats = BacktestEngine._compute_stats(portfolio.completed_trades)
        return portfolio, stats

    @classmethod
    def _close_position(
        cls,
        portfolio: PortfolioState,
        pos: OpenPosition,
        exit_price: float,
        exit_reason: str,
        exit_date_str: str,
    ):
        """Closes remaining shares of an OpenPosition and credits net proceeds to available cash."""
        rem_shares = pos.remaining_shares
        if rem_shares <= 0:
            return

        exit_cost = cls.calculate_exit_friction(exit_price, rem_shares)
        exit_proceeds = (exit_price * rem_shares) - exit_cost

        # Return net proceeds to available cash and reduce invested capital
        portfolio.cash_available += exit_proceeds
        portfolio.invested_capital -= (pos.entry_price * rem_shares)

        final_leg_pnl = (exit_price - pos.entry_price) * rem_shares - pos.entry_cost - exit_cost
        pos.realized_pnl_so_far += final_leg_pnl
        portfolio.realized_pnl += final_leg_pnl
        pos.executed_sell_value += (exit_price * rem_shares)
        pos.remaining_shares = 0

        # Calculate total round-trip friction costs
        total_costs = pos.entry_cost + exit_cost
        net_trade_pnl = pos.realized_pnl_so_far
        gross_trade_pnl = net_trade_pnl + total_costs
        pnl_pct = (net_trade_pnl / (pos.entry_price * pos.shares)) * 100.0 if (pos.entry_price * pos.shares) > 0 else 0.0

        mae_pct = ((pos.min_low - pos.entry_price) / pos.entry_price) * 100.0
        mfe_pct = ((pos.max_high - pos.entry_price) / pos.entry_price) * 100.0

        completed_trade = BacktestTrade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            stop_loss=pos.stop_loss,
            target_1=pos.target_1,
            target_2=pos.target_2,
            target_3=pos.target_3,
            shares=pos.shares,
            exit_date=exit_date_str,
            exit_price=round(exit_price, 2),
            exit_reason=exit_reason,
            pnl_pct=round(pnl_pct, 2),
            pnl_rupees=round(net_trade_pnl, 2),
            gross_pnl_rupees=round(gross_trade_pnl, 2),
            transaction_cost_rupees=round(total_costs, 2),
            holding_sessions=pos.holding_sessions,
            max_adverse_excursion_pct=round(mae_pct, 2),
            max_favorable_excursion_pct=round(mfe_pct, 2),
            executed_buy_value=round(pos.entry_price * pos.shares, 2),
            executed_sell_value=round(pos.executed_sell_value, 2),
        )
        portfolio.completed_trades.append(completed_trade)
