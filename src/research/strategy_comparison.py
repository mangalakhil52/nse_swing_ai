"""Historical baseline-vs-alpha strategy experiment.

The experiment deliberately keeps the baseline and enhanced strategies identical
except for the P1 alpha gate. This makes attribution measurable instead of
confounding multiple changes at once.

All signal features are computed from bars available at the signal timestamp.
Future bars are used only by BacktestEngine after the signal has been frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Any

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestTrade
from src.backtest.backtest_metrics import max_drawdown, profit_factor, sharpe, sortino
from src.core.types import MarketRegime
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
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"])
    return pd.to_datetime(df.index)


def generate_signal_sets(
    stock_dfs: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    config: ExperimentConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate matched baseline/enhanced entry events without future leakage."""
    cfg = config or ExperimentConfig()
    baseline: list[dict[str, Any]] = []
    enhanced: list[dict[str, Any]] = []

    # First pass: identify deterministic pattern events. Indicators are computed
    # once on the full history because each indicator is causal; at event i only
    # rows <= i are passed into the pattern detector.
    candidates_by_date: dict[str, list[dict[str, Any]]] = {}
    for symbol, raw in stock_dfs.items():
        if raw is None or len(raw) < cfg.min_history_bars:
            continue
        df = raw.copy().sort_values("timestamp") if "timestamp" in raw.columns else raw.copy()
        df = TechnicalIndicators.compute_all_indicators(df)
        for i in range(cfg.min_history_bars - 1, len(df)):
            pit = df.iloc[: i + 1].copy()
            matches = PatternRecognizer.evaluate_all_patterns(pit)
            match = matches[0] if matches and matches[0].quality_score >= cfg.min_pattern_quality else None
            if match is None:
                continue
            event_date = _dates(df).iloc[i].strftime("%Y-%m-%d")
            candidates_by_date.setdefault(event_date, []).append({
                "symbol": symbol,
                "entry_idx": i,
                "event_date": event_date,
                "pattern": match.pattern_type.value,
                "pattern_quality": float(match.quality_score),
            })

    for event_date, candidates in sorted(candidates_by_date.items()):
        for candidate in candidates:
            baseline.append(candidate.copy())

        # Cross-sectional ranking is restricted to the candidates that actually
        # triggered the identical baseline setup on this date. No future bars are
        # exposed to alpha calculation.
        pit_frames: dict[str, pd.DataFrame] = {}
        for candidate in candidates:
            symbol = candidate["symbol"]
            raw = stock_dfs[symbol].copy().sort_values("timestamp")
            idx = int(candidate["entry_idx"])
            pit_frames[symbol] = raw.iloc[: idx + 1]
        bench = benchmark_df.copy()
        if "timestamp" in bench.columns:
            bench = bench[pd.to_datetime(bench["timestamp"]) <= pd.Timestamp(event_date)]
        # Raw alpha is used here as a screening gate. Full regime-conditioned
        # cross-sectional scoring is exposed separately and can be substituted
        # once historical regime snapshots are supplied.
        for candidate in candidates:
            symbol = candidate["symbol"]
            try:
                alpha = compute_alpha_features(pit_frames[symbol], bench).alpha_score
            except (ValueError, KeyError, TypeError):
                alpha = float("nan")
            if pd.notna(alpha) and float(alpha) >= cfg.min_alpha_score:
                row = candidate.copy()
                row["alpha_score"] = float(alpha)
                enhanced.append(row)

    return baseline[: cfg.max_signals_per_symbol * max(1, len(stock_dfs))], enhanced[: cfg.max_signals_per_symbol * max(1, len(stock_dfs))]


def _run_trades(signals: list[dict[str, Any]], stock_dfs: dict[str, pd.DataFrame]) -> list[BacktestTrade]:
    trades: list[BacktestTrade] = []
    last_exit_by_symbol: dict[str, int] = {}
    for signal in sorted(signals, key=lambda x: (x["event_date"], x["symbol"], x["entry_idx"])):
        symbol = signal["symbol"]
        df = stock_dfs.get(symbol)
        if df is None:
            continue
        idx = int(signal["entry_idx"])
        if idx <= last_exit_by_symbol.get(symbol, -1):
            continue
        trade, _ = BacktestEngine.backtest_entry_signal(symbol, df, idx)
        if trade is None:
            continue
        trades.append(trade)
        exit_ts = pd.Timestamp(trade.exit_date) if trade.exit_date else pd.Timestamp(signal["event_date"])
        dates = _dates(df)
        matches = dates[dates <= exit_ts]
        last_exit_by_symbol[symbol] = int(matches.index[-1]) if len(matches) else idx
    return trades


def summarize(variant: str, signals: list[dict[str, Any]], trades: list[BacktestTrade]) -> ExperimentReport:
    returns = [float(t.pnl_pct or 0.0) / 100.0 for t in trades]
    mfes = [float(t.max_favorable_excursion_pct or 0.0) for t in trades]
    maes = [float(t.max_adverse_excursion_pct or 0.0) for t in trades]
    wins = [r for r in returns if r > 0]
    return ExperimentReport(
        variant=variant,
        signals=len(signals),
        trades=len(trades),
        win_rate_pct=round(100 * len(wins) / len(returns), 2) if returns else 0.0,
        profit_factor=round(profit_factor(returns), 3) if returns else 0.0,
        expectancy_pct=round(100 * (sum(returns) / len(returns)), 4) if returns else 0.0,
        sharpe=round(sharpe(returns), 3) if returns else 0.0,
        sortino=round(sortino(returns), 3) if returns else 0.0,
        max_drawdown_pct=round(100 * max_drawdown(returns), 3) if returns else 0.0,
        total_return_pct=round(100 * sum(returns), 3),
        avg_mfe_pct=round(sum(mfes) / len(mfes), 3) if mfes else 0.0,
        avg_mae_pct=round(sum(maes) / len(maes), 3) if maes else 0.0,
    )


def compare(
    stock_dfs: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    baseline, enhanced = generate_signal_sets(stock_dfs, benchmark_df, config)
    base_trades = _run_trades(baseline, stock_dfs)
    enhanced_trades = _run_trades(enhanced, stock_dfs)
    return {
        "baseline": asdict(summarize("BASELINE", baseline, base_trades)),
        "enhanced": asdict(summarize("P1_ALPHA_GATE", enhanced, enhanced_trades)),
        "delta": {
            "trades": len(enhanced_trades) - len(base_trades),
            "win_rate_pct": summarize("E", enhanced, enhanced_trades).win_rate_pct - summarize("B", baseline, base_trades).win_rate_pct,
            "profit_factor": summarize("E", enhanced, enhanced_trades).profit_factor - summarize("B", baseline, base_trades).profit_factor,
            "sharpe": summarize("E", enhanced, enhanced_trades).sharpe - summarize("B", baseline, base_trades).sharpe,
            "max_drawdown_pct": summarize("E", enhanced, enhanced_trades).max_drawdown_pct - summarize("B", baseline, base_trades).max_drawdown_pct,
        },
    }
