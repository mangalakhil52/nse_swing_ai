"""
Portfolio Performance Analytics Module — src/backtest/performance.py (P0 Fix #11C)

Builds a rigorous performance-analysis layer on top of the chronological PortfolioBacktestEngine
and PortfolioState equity curve.

Calculates:
  1. Chronological Equity Curve & Mark-to-Market Accounting.
  2. Return Metrics (Total Return, CAGR, Annualized Return).
  3. Drawdown Metrics (Series, Max Drawdown %, Max Drawdown Rupees, Duration, Peak, Trough, Recovery dates).
  4. Risk-Adjusted Performance (Annualized Volatility, Sharpe Ratio, Sortino Ratio).
  5. Trade Statistics (Win Rate, Gross Profit/Loss, Profit Factor, Expectancy, Winner/Loser Averages).
  6. R-Multiple Analysis (Realized R, R-Distribution Buckets).
  7. Holding Period Statistics (Average, Median, Min, Max holding days).
  8. Portfolio Exposure & Position Metrics (Max/Avg Exposure %, Min/Avg Cash %, Max/Avg Open Positions).
  9. Portfolio Turnover (Buy Value, Sell Value, Total Turnover, Turnover %).
 10. Transaction Cost Impact (Gross PnL - Friction Costs = Net PnL).
 11. Benchmark Analysis (Optional NIFTY 50 benchmark comparison, excess return, benchmark drawdown).
"""

from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np
import pandas as pd

from src.backtest.portfolio import DailyPortfolioSnapshot, PortfolioState, OpenPosition
from src.backtest.engine import BacktestTrade


@dataclass
class ReturnMetrics:
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    annualized_return_pct: float = 0.0
    elapsed_days: float = 0.0
    trading_days: int = 0


@dataclass
class DrawdownMetrics:
    max_drawdown_rupees: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    peak_date: str | None = None
    trough_date: str | None = None
    recovery_date: str | None = None
    drawdown_series_pct: list[float] = field(default_factory=list)


@dataclass
class RiskMetrics:
    annualized_volatility_pct: float = 0.0
    risk_free_rate_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0


@dataclass
class TradeMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    average_winner: float = 0.0
    average_loser: float = 0.0
    expectancy_per_trade: float = 0.0
    average_trade_pnl: float = 0.0
    median_trade_pnl: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    average_holding_period: float = 0.0
    median_holding_period: float = 0.0


@dataclass
class RMultipleMetrics:
    average_R: float = 0.0
    median_R: float = 0.0
    winning_R_average: float = 0.0
    losing_R_average: float = 0.0
    distribution: dict[str, int] = field(default_factory=dict)
    distribution_pct: dict[str, float] = field(default_factory=dict)


@dataclass
class HoldingPeriodMetrics:
    average_holding_days: float = 0.0
    median_holding_days: float = 0.0
    max_holding_days: int = 0
    min_holding_days: int = 0


@dataclass
class ExposureMetrics:
    maximum_exposure_pct: float = 0.0
    average_exposure_pct: float = 0.0
    minimum_cash_pct: float = 0.0
    average_cash_pct: float = 0.0
    maximum_open_positions: int = 0
    average_open_positions: float = 0.0


@dataclass
class TurnoverMetrics:
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    total_turnover: float = 0.0
    turnover_pct: float = 0.0


@dataclass
class TransactionCostMetrics:
    gross_pnl: float = 0.0
    total_transaction_costs: float = 0.0
    net_pnl: float = 0.0


@dataclass
class BenchmarkMetrics:
    benchmark_symbol: str = "NIFTY 50"
    status: str = "UNAVAILABLE: Benchmark data not provided."
    benchmark_initial_value: float = 0.0
    benchmark_final_value: float = 0.0
    benchmark_total_return_pct: float = 0.0
    benchmark_cagr_pct: float = 0.0
    benchmark_max_drawdown_pct: float = 0.0
    strategy_excess_return_pct: float = 0.0


@dataclass
class PerformanceReport:
    status: str = "OK"
    return_metrics: ReturnMetrics = field(default_factory=ReturnMetrics)
    drawdown_metrics: DrawdownMetrics = field(default_factory=DrawdownMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    trade_metrics: TradeMetrics = field(default_factory=TradeMetrics)
    r_metrics: RMultipleMetrics = field(default_factory=RMultipleMetrics)
    holding_period_metrics: HoldingPeriodMetrics = field(default_factory=HoldingPeriodMetrics)
    exposure_metrics: ExposureMetrics = field(default_factory=ExposureMetrics)
    turnover_metrics: TurnoverMetrics = field(default_factory=TurnoverMetrics)
    transaction_cost_metrics: TransactionCostMetrics = field(default_factory=TransactionCostMetrics)
    benchmark_metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)


class PerformanceAnalyzer:
    """
    Downstream performance analysis consumer. Performs zero state modifications.
    """

    @classmethod
    def analyze_portfolio(
        cls,
        portfolio: PortfolioState,
        benchmark_df: pd.DataFrame | None = None,
        risk_free_rate_pct: float = 0.0,
    ) -> PerformanceReport:
        """
        Generates a comprehensive structured PerformanceReport from PortfolioState and completed trades.
        Fail-closed: Gracefully handles empty portfolio equity curves, zero initial capital, and missing benchmark data.
        """
        if portfolio is None or portfolio.initial_capital <= 0.0:
            return PerformanceReport(status="INVALID_CAPITAL")

        if not portfolio.equity_curve and not portfolio.completed_trades:
            return PerformanceReport(
                status="EMPTY_BACKTEST",
                return_metrics=ReturnMetrics(
                    initial_capital=portfolio.initial_capital,
                    final_equity=portfolio.total_equity,
                ),
            )

        ret_m = cls._compute_return_metrics(portfolio)
        dd_m = cls._compute_drawdown_metrics(portfolio)
        risk_m = cls._compute_risk_metrics(portfolio, risk_free_rate_pct, ret_m)
        trade_m = cls._compute_trade_metrics(portfolio)
        r_m = cls._compute_r_metrics(portfolio)
        hp_m = cls._compute_holding_period_metrics(portfolio)
        exp_m = cls._compute_exposure_metrics(portfolio)
        turn_m = cls._compute_turnover_metrics(portfolio)
        cost_m = cls._compute_transaction_cost_metrics(portfolio)
        bench_m = cls._compute_benchmark_metrics(portfolio, benchmark_df, ret_m.total_return_pct)

        status_str = "OK" if portfolio.completed_trades or portfolio.equity_curve else "EMPTY_BACKTEST"

        return PerformanceReport(
            status=status_str,
            return_metrics=ret_m,
            drawdown_metrics=dd_m,
            risk_metrics=risk_m,
            trade_metrics=trade_m,
            r_metrics=r_m,
            holding_period_metrics=hp_m,
            exposure_metrics=exp_m,
            turnover_metrics=turn_m,
            transaction_cost_metrics=cost_m,
            benchmark_metrics=bench_m,
        )

    @classmethod
    def _compute_return_metrics(cls, portfolio: PortfolioState) -> ReturnMetrics:
        initial = portfolio.initial_capital
        final = portfolio.total_equity
        if initial <= 0.0:
            return ReturnMetrics()

        total_return_pct = round(((final / initial) - 1.0) * 100.0, 4)

        curve = portfolio.equity_curve
        trading_days = len(curve)
        if trading_days >= 2:
            d1 = pd.to_datetime(curve[0].date)
            d2 = pd.to_datetime(curve[-1].date)
            elapsed_days = float((d2 - d1).days)
        else:
            elapsed_days = 0.0

        if initial > 0.0 and final > 0.0 and elapsed_days > 0.0:
            cagr_pct = round((((final / initial) ** (365.25 / elapsed_days)) - 1.0) * 100.0, 4)
        else:
            cagr_pct = 0.0

        if initial > 0.0 and final > 0.0 and trading_days > 0:
            annualized_return_pct = round((((final / initial) ** (252.0 / trading_days)) - 1.0) * 100.0, 4)
        else:
            annualized_return_pct = 0.0

        return ReturnMetrics(
            initial_capital=round(initial, 2),
            final_equity=round(final, 2),
            total_return_pct=total_return_pct,
            cagr_pct=cagr_pct,
            annualized_return_pct=annualized_return_pct,
            elapsed_days=elapsed_days,
            trading_days=trading_days,
        )

    @classmethod
    def _compute_drawdown_metrics(cls, portfolio: PortfolioState) -> DrawdownMetrics:
        curve = portfolio.equity_curve
        if not curve:
            return DrawdownMetrics()

        running_peak = -1.0
        drawdown_rupees_list: list[float] = []
        drawdown_pct_list: list[float] = []

        for eq in curve:
            if eq.total_equity > running_peak:
                running_peak = eq.total_equity
            dd_r = running_peak - eq.total_equity
            dd_p = ((eq.total_equity - running_peak) / running_peak) * 100.0 if running_peak > 0.0 else 0.0
            drawdown_rupees_list.append(round(dd_r, 2))
            drawdown_pct_list.append(round(dd_p, 4))

        max_dd_r = max(drawdown_rupees_list) if drawdown_rupees_list else 0.0
        max_dd_p = min(drawdown_pct_list) if drawdown_pct_list else 0.0

        # Locate trough index for worst drawdown
        idx_trough = int(np.argmin(drawdown_pct_list)) if drawdown_pct_list else 0
        trough_date = curve[idx_trough].date if curve else None

        # Peak date preceding trough
        running_max_val = -1.0
        idx_peak = 0
        for i in range(idx_trough + 1):
            if curve[i].total_equity >= running_max_val:
                running_max_val = curve[i].total_equity
                idx_peak = i

        peak_date = curve[idx_peak].date if curve else None
        peak_val = curve[idx_peak].total_equity if curve else 0.0

        # Recovery date
        recovery_date = None
        for i in range(idx_trough + 1, len(curve)):
            if curve[i].total_equity >= peak_val:
                recovery_date = curve[i].date
                break

        if peak_date:
            if recovery_date:
                duration_days = (pd.to_datetime(recovery_date) - pd.to_datetime(peak_date)).days
            else:
                duration_days = (pd.to_datetime(curve[-1].date) - pd.to_datetime(peak_date)).days
        else:
            duration_days = 0

        return DrawdownMetrics(
            max_drawdown_rupees=round(max_dd_r, 2),
            max_drawdown_pct=round(max_dd_p, 4),
            max_drawdown_duration_days=duration_days,
            peak_date=peak_date,
            trough_date=trough_date,
            recovery_date=recovery_date,
            drawdown_series_pct=drawdown_pct_list,
        )

    @classmethod
    def _compute_risk_metrics(
        cls,
        portfolio: PortfolioState,
        risk_free_rate_pct: float,
        ret_metrics: ReturnMetrics,
    ) -> RiskMetrics:
        """
        Calculates Sharpe and Sortino ratios using daily portfolio excess returns.

        Formula Conventions:
          rf_daily = (1 + risk_free_rate_pct / 100)^(1 / 252) - 1
          daily_excess_return[t] = daily_portfolio_return[t] - rf_daily
          mean_daily_excess = mean(daily_excess_return)

          Sharpe Ratio = (mean_daily_excess / std(daily_excess_return, ddof=1)) * sqrt(252)
          Downside Deviation (daily) = sqrt(mean(min(daily_excess_return, 0)^2))
          Sortino Ratio = (mean_daily_excess / downside_deviation_daily) * sqrt(252)
        """
        curve = portfolio.equity_curve
        if len(curve) < 2:
            return RiskMetrics(risk_free_rate_pct=risk_free_rate_pct)

        equity_series = [eq.total_equity for eq in curve]
        daily_returns = [
            (equity_series[i] - equity_series[i - 1]) / equity_series[i - 1]
            for i in range(1, len(equity_series))
            if equity_series[i - 1] > 0.0
        ]

        if not daily_returns:
            return RiskMetrics(risk_free_rate_pct=risk_free_rate_pct)

        rf_daily = ((1.0 + (risk_free_rate_pct / 100.0)) ** (1.0 / 252.0)) - 1.0
        daily_excess = [r - rf_daily for r in daily_returns]
        mean_daily_excess = float(np.mean(daily_excess))

        daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
        annualized_vol_pct = round(daily_std * np.sqrt(252) * 100.0, 4)

        daily_excess_std = float(np.std(daily_excess, ddof=1)) if len(daily_excess) > 1 else 0.0
        if daily_excess_std > 0.0:
            sharpe = round((mean_daily_excess / daily_excess_std) * np.sqrt(252), 4)
        else:
            sharpe = 0.0

        downside_obs = [min(x, 0.0) for x in daily_excess]
        downside_sq_mean = float(np.mean([d ** 2 for d in downside_obs])) if downside_obs else 0.0
        daily_downside_dev = np.sqrt(downside_sq_mean)

        if daily_downside_dev > 0.0:
            sortino = round((mean_daily_excess / daily_downside_dev) * np.sqrt(252), 4)
        else:
            sortino = 0.0

        return RiskMetrics(
            annualized_volatility_pct=annualized_vol_pct,
            risk_free_rate_pct=risk_free_rate_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
        )

    @classmethod
    def _compute_trade_metrics(cls, portfolio: PortfolioState) -> TradeMetrics:
        completed = portfolio.completed_trades
        total_trades = len(completed)
        if total_trades == 0:
            return TradeMetrics()

        winners = [t for t in completed if t.pnl_rupees > 0.0]
        losers = [t for t in completed if t.pnl_rupees < 0.0]
        breakevens = [t for t in completed if t.pnl_rupees == 0.0]

        winning_trades = len(winners)
        losing_trades = len(losers)
        breakeven_trades = len(breakevens)

        win_rate_pct = round((winning_trades / total_trades) * 100.0, 2)
        gross_profit = round(sum(t.pnl_rupees for t in winners), 2)
        gross_loss = round(abs(sum(t.pnl_rupees for t in losers)), 2)

        if gross_loss > 0.0:
            profit_factor = round(gross_profit / gross_loss, 4)
        elif gross_profit > 0.0:
            profit_factor = 100.0
        else:
            profit_factor = 0.0

        avg_winner = round(gross_profit / winning_trades, 2) if winning_trades > 0 else 0.0
        avg_loser = round(gross_loss / losing_trades, 2) if losing_trades > 0 else 0.0

        all_pnls = [t.pnl_rupees for t in completed]
        expectancy = round(sum(all_pnls) / total_trades, 2)
        avg_pnl = round(float(np.mean(all_pnls)), 2)
        median_pnl = round(float(np.median(all_pnls)), 2)

        largest_win = round(max(all_pnls), 2)
        largest_loss = round(min(all_pnls), 2)

        holdings = [t.holding_sessions for t in completed]
        avg_holding = round(float(np.mean(holdings)), 2)
        median_holding = round(float(np.median(holdings)), 2)

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate_pct=win_rate_pct,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            average_winner=avg_winner,
            average_loser=avg_loser,
            expectancy_per_trade=expectancy,
            average_trade_pnl=avg_pnl,
            median_trade_pnl=median_pnl,
            largest_winner=largest_win,
            largest_loser=largest_loss,
            average_holding_period=avg_holding,
            median_holding_period=median_holding,
        )

    @classmethod
    def _compute_r_metrics(cls, portfolio: PortfolioState) -> RMultipleMetrics:
        completed = portfolio.completed_trades
        if not completed:
            return RMultipleMetrics()

        r_values: list[float] = []
        win_r: list[float] = []
        lose_r: list[float] = []

        dist = {
            "R < -1": 0,
            "-1 <= R < 0": 0,
            "0 <= R < 1": 0,
            "1 <= R < 2": 0,
            "2 <= R < 3": 0,
            "R >= 3": 0,
        }

        for t in completed:
            init_risk = (t.entry_price - t.stop_loss) * t.shares
            r_mult = t.pnl_rupees / init_risk if init_risk > 0.0 else 0.0
            r_mult_rounded = round(r_mult, 4)
            r_values.append(r_mult_rounded)

            if t.pnl_rupees > 0.0:
                win_r.append(r_mult_rounded)
            elif t.pnl_rupees < 0.0:
                lose_r.append(r_mult_rounded)

            if r_mult_rounded < -1.0:
                dist["R < -1"] += 1
            elif -1.0 <= r_mult_rounded < 0.0:
                dist["-1 <= R < 0"] += 1
            elif 0.0 <= r_mult_rounded < 1.0:
                dist["0 <= R < 1"] += 1
            elif 1.0 <= r_mult_rounded < 2.0:
                dist["1 <= R < 2"] += 1
            elif 2.0 <= r_mult_rounded < 3.0:
                dist["2 <= R < 3"] += 1
            else:
                dist["R >= 3"] += 1

        n = len(completed)
        dist_pct = {k: round((v / n) * 100.0, 2) for k, v in dist.items()}

        avg_R = round(float(np.mean(r_values)), 4) if r_values else 0.0
        med_R = round(float(np.median(r_values)), 4) if r_values else 0.0
        win_R_avg = round(float(np.mean(win_r)), 4) if win_r else 0.0
        lose_R_avg = round(float(np.mean(lose_r)), 4) if lose_r else 0.0

        return RMultipleMetrics(
            average_R=avg_R,
            median_R=med_R,
            winning_R_average=win_R_avg,
            losing_R_average=lose_R_avg,
            distribution=dist,
            distribution_pct=dist_pct,
        )

    @classmethod
    def _compute_holding_period_metrics(cls, portfolio: PortfolioState) -> HoldingPeriodMetrics:
        completed = portfolio.completed_trades
        if not completed:
            return HoldingPeriodMetrics()

        calendar_days: list[int] = []
        for t in completed:
            if t.entry_date and t.exit_date:
                days = (pd.to_datetime(t.exit_date) - pd.to_datetime(t.entry_date)).days
                calendar_days.append(max(0, days))
            else:
                calendar_days.append(t.holding_sessions)

        avg_days = round(float(np.mean(calendar_days)), 2)
        med_days = round(float(np.median(calendar_days)), 2)
        max_days = int(max(calendar_days))
        min_days = int(min(calendar_days))

        return HoldingPeriodMetrics(
            average_holding_days=avg_days,
            median_holding_days=med_days,
            max_holding_days=max_days,
            min_holding_days=min_days,
        )

    @classmethod
    def _compute_exposure_metrics(cls, portfolio: PortfolioState) -> ExposureMetrics:
        curve = portfolio.equity_curve
        if not curve:
            return ExposureMetrics()

        exposures = [eq.exposure_pct for eq in curve]
        cash_pcts = [
            (eq.cash_available / eq.total_equity) * 100.0 if eq.total_equity > 0.0 else 0.0
            for eq in curve
        ]
        open_pos_counts = [eq.open_positions for eq in curve]

        max_exp = round(float(max(exposures)), 2)
        avg_exp = round(float(np.mean(exposures)), 2)
        min_cash = round(float(min(cash_pcts)), 2)
        avg_cash = round(float(np.mean(cash_pcts)), 2)

        max_pos = int(max(open_pos_counts))
        avg_pos = round(float(np.mean(open_pos_counts)), 2)

        return ExposureMetrics(
            maximum_exposure_pct=max_exp,
            average_exposure_pct=avg_exp,
            minimum_cash_pct=min_cash,
            average_cash_pct=avg_cash,
            maximum_open_positions=max_pos,
            average_open_positions=avg_pos,
        )

    @classmethod
    def _compute_turnover_metrics(cls, portfolio: PortfolioState) -> TurnoverMetrics:
        """
        Calculates portfolio turnover based on actual executed buy and sell notionals.
        Rejected signals contribute ZERO to turnover.
        Partial exits are counted based on executed sell notional for each leg.
        """
        completed = portfolio.completed_trades
        open_positions = portfolio.open_positions

        buy_val = sum(
            t.executed_buy_value if getattr(t, "executed_buy_value", 0.0) > 0.0 else (t.entry_price * t.shares)
            for t in completed
        ) + sum(pos.entry_price * pos.shares for pos in open_positions.values())

        sell_val = sum(
            t.executed_sell_value if getattr(t, "executed_sell_value", 0.0) > 0.0 else ((t.exit_price or t.entry_price) * t.shares)
            for t in completed
        ) + sum(pos.executed_sell_value for pos in open_positions.values())

        total_turnover = round(buy_val + sell_val, 2)
        initial = portfolio.initial_capital
        turnover_pct = round((total_turnover / initial) * 100.0, 2) if initial > 0.0 else 0.0

        return TurnoverMetrics(
            total_buy_value=round(buy_val, 2),
            total_sell_value=round(sell_val, 2),
            total_turnover=total_turnover,
            turnover_pct=turnover_pct,
        )

    @classmethod
    def _compute_transaction_cost_metrics(cls, portfolio: PortfolioState) -> TransactionCostMetrics:
        completed = portfolio.completed_trades
        gross = sum(t.gross_pnl_rupees for t in completed)
        costs = sum(t.transaction_cost_rupees for t in completed)
        net = sum(t.pnl_rupees for t in completed)

        return TransactionCostMetrics(
            gross_pnl=round(gross, 2),
            total_transaction_costs=round(costs, 2),
            net_pnl=round(net, 2),
        )

    @classmethod
    def _compute_benchmark_metrics(
        cls,
        portfolio: PortfolioState,
        benchmark_df: pd.DataFrame | None,
        strategy_total_return_pct: float,
    ) -> BenchmarkMetrics:
        if benchmark_df is None or benchmark_df.empty or "close" not in benchmark_df.columns:
            return BenchmarkMetrics(
                status="UNAVAILABLE: Benchmark data not provided."
            )

        b_df = benchmark_df.copy()
        if "timestamp" in b_df.columns:
            b_df["date_str"] = pd.to_datetime(b_df["timestamp"]).dt.strftime("%Y-%m-%d")
        else:
            b_df["date_str"] = pd.to_datetime(b_df.index).strftime("%Y-%m-%d")

        curve_dates = [eq.date for eq in portfolio.equity_curve]
        if not curve_dates:
            return BenchmarkMetrics(status="UNAVAILABLE: Empty strategy equity curve.")

        matched = b_df[b_df["date_str"].isin(curve_dates)]
        if matched.empty or len(matched) < 2:
            return BenchmarkMetrics(status="UNAVAILABLE: Insufficient benchmark date overlap.")

        first_val = float(matched.iloc[0]["close"])
        last_val = float(matched.iloc[-1]["close"])
        if first_val <= 0.0:
            return BenchmarkMetrics(status="UNAVAILABLE: Invalid benchmark prices.")

        total_ret = round(((last_val / first_val) - 1.0) * 100.0, 4)
        elapsed_days = float((pd.to_datetime(matched.iloc[-1]["date_str"]) - pd.to_datetime(matched.iloc[0]["date_str"])).days)

        if elapsed_days > 0.0 and last_val > 0.0:
            cagr = round((((last_val / first_val) ** (365.25 / elapsed_days)) - 1.0) * 100.0, 4)
        else:
            cagr = 0.0

        # Benchmark Max Drawdown
        b_closes = matched["close"].tolist()
        b_peak = -1.0
        b_drawdowns = []
        for val in b_closes:
            if val > b_peak:
                b_peak = val
            dd = ((val - b_peak) / b_peak) * 100.0 if b_peak > 0.0 else 0.0
            b_drawdowns.append(dd)
        max_b_dd = round(min(b_drawdowns), 4) if b_drawdowns else 0.0

        excess_ret = round(strategy_total_return_pct - total_ret, 4)

        return BenchmarkMetrics(
            benchmark_symbol="NIFTY 50",
            status="OK",
            benchmark_initial_value=round(first_val, 2),
            benchmark_final_value=round(last_val, 2),
            benchmark_total_return_pct=total_ret,
            benchmark_cagr_pct=cagr,
            benchmark_max_drawdown_pct=max_b_dd,
            strategy_excess_return_pct=excess_ret,
        )
