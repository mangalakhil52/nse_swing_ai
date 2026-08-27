"""Historical baseline-vs-alpha strategy experiment.

Baseline and enhanced variants share the same deterministic setup and trade
construction. The only strategy difference is the P1 alpha gate. Features are
computed from bars available at the signal timestamp; future bars are consumed
only by the backtest execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestTrade
from src.quant.backtest_metrics import max_drawdown, profit_factor, sharpe, sortino
from src.quant.advanced_alpha import compute_alpha_features
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer


@dataclass(frozen=True)
class ExperimentConfig:
    min_history_bars: int = 130
    min_pattern_quality: float = 75.0
    min_alpha_score: float = 0.10
    max_signals_per_symbol: int = 1000


@dataclass(frozen=True)
class ExperimentReport:
    variant: str
    signals: int
    trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    total_return_pct: float
    avg_mfe_pct: float
    avg_mae_pct: float


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.to_datetime(df.index)


def generate_signal_sets(stock_dfs: dict[str, pd.DataFrame], benchmark_df: pd.DataFrame, config: ExperimentConfig | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate matched baseline/P1 entry events with point-in-time features."""
    cfg = config or ExperimentConfig()
    baseline: list[dict[str, Any]] = []
    enhanced: list[dict[str, Any]] = []
    candidates_by_date: dict[str, list[dict[str, Any]]] = {}

    for symbol, raw in stock_dfs.items():
        if raw is None or len(raw) < cfg.min_history_bars:
            continue
        df = raw.copy().sort_values("timestamp") if "timestamp" in raw.columns else raw.copy()
        df = TechnicalIndicators.compute_all_indicators(df)
        dates = _dates(df)
        for i in range(cfg.min_history_bars - 1, len(df)):
            matches = PatternRecognizer.evaluate_all_patterns(df.iloc[: i + 1])
            if not matches or matches[0].quality_score < cfg.min_pattern_quality:
                continue
            event_date = dates.iloc[i].strftime("%Y-%m-%d")
            candidates_by_date.setdefault(event_date, []).append({
                "symbol": symbol, "entry_idx": i, "event_date": event_date,
                "pattern": matches[0].pattern_type.value,
                "pattern_quality": float(matches[0].quality_score),
            })

    for event_date, candidates in sorted(candidates_by_date.items()):
        baseline.extend(c.copy() for c in candidates)
        for candidate in candidates:
            symbol = candidate["symbol"]
            raw = stock_dfs[symbol].copy().sort_values("timestamp") if "timestamp" in stock_dfs[symbol].columns else stock_dfs[symbol].copy()
            pit = raw.iloc[: int(candidate["entry_idx"]) + 1]
            bench = benchmark_df.copy()
            if "timestamp" in bench.columns:
                bench = bench[pd.to_datetime(bench["timestamp"]) <= pd.Timestamp(event_date)]
            try:
                alpha = compute_alpha_features(pit, bench).alpha_score
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(alpha) and alpha >= cfg.min_alpha_score:
                row = candidate.copy()
                row["alpha_score"] = float(alpha)
                enhanced.append(row)

    limit = cfg.max_signals_per_symbol * max(1, len(stock_dfs))
    return baseline[:limit], enhanced[:limit]


def _run_trades(signals: list[dict[str, Any]], stock_dfs: dict[str, pd.DataFrame]) -> list[BacktestTrade]:
    """Execute signals chronologically and prevent overlapping positions per symbol."""
    trades: list[BacktestTrade] = []
    next_entry_idx: dict[str, int] = {}
    for signal in sorted(signals, key=lambda x: (x["event_date"], x["symbol"], x["entry_idx"])):
        symbol = signal["symbol"]
        df = stock_dfs.get(symbol)
        if df is None:
            continue
        idx = int(signal["entry_idx"])
        if idx < next_entry_idx.get(symbol, 0):
            continue
        trade, _ = BacktestEngine.backtest_entry_signal(symbol, df, idx)
        if trade is None:
            continue
        trades.append(trade)
        exit_ts = pd.Timestamp(trade.exit_date) if trade.exit_date else pd.Timestamp(signal["event_date"])
        dates = _dates(df).reset_index(drop=True)
        next_entry_idx[symbol] = int(np.searchsorted(dates.to_numpy(), exit_ts.to_datetime64(), side="right"))
    return trades


def summarize(variant: str, signals: list[dict[str, Any]], trades: list[BacktestTrade]) -> ExperimentReport:
    returns = [float(t.pnl_pct or 0.0) / 100.0 for t in trades]
    mfes = [float(t.max_favorable_excursion_pct or 0.0) for t in trades]
    maes = [float(t.max_adverse_excursion_pct or 0.0) for t in trades]
    wins = [r for r in returns if r > 0]
    return ExperimentReport(
        variant=variant, signals=len(signals), trades=len(trades),
        win_rate_pct=round(100 * len(wins) / len(returns), 2) if returns else 0.0,
        profit_factor=round(profit_factor(returns), 3) if returns else 0.0,
        expectancy_pct=round(100 * np.mean(returns), 4) if returns else 0.0,
        sharpe=round(sharpe(returns), 3) if returns else 0.0,
        sortino=round(sortino(returns), 3) if returns else 0.0,
        max_drawdown_pct=round(100 * max_drawdown(returns), 3) if returns else 0.0,
        total_return_pct=round(100 * sum(returns), 3),
        avg_mfe_pct=round(float(np.mean(mfes)), 3) if mfes else 0.0,
        avg_mae_pct=round(float(np.mean(maes)), 3) if maes else 0.0,
    )


def compare(stock_dfs: dict[str, pd.DataFrame], benchmark_df: pd.DataFrame, config: ExperimentConfig | None = None) -> dict[str, Any]:
    baseline, enhanced = generate_signal_sets(stock_dfs, benchmark_df, config)
    base_report = summarize("BASELINE", baseline, _run_trades(baseline, stock_dfs))
    enhanced_report = summarize("P1_ALPHA_GATE", enhanced, _run_trades(enhanced, stock_dfs))
    return {
        "baseline": asdict(base_report), "enhanced": asdict(enhanced_report),
        "delta": {
            "signals": enhanced_report.signals - base_report.signals,
            "trades": enhanced_report.trades - base_report.trades,
            "win_rate_pct": round(enhanced_report.win_rate_pct - base_report.win_rate_pct, 3),
            "profit_factor": round(enhanced_report.profit_factor - base_report.profit_factor, 3),
            "expectancy_pct": round(enhanced_report.expectancy_pct - base_report.expectancy_pct, 4),
            "sharpe": round(enhanced_report.sharpe - base_report.sharpe, 3),
            "sortino": round(enhanced_report.sortino - base_report.sortino, 3),
            "max_drawdown_pct": round(enhanced_report.max_drawdown_pct - base_report.max_drawdown_pct, 3),
        },
    }
