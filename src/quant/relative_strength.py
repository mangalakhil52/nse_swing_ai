"""
Relative Strength (RS) Computation Engine Module.
Calculates Mansfield Relative Strength against NIFTY 50 and benchmark Sector indices.
Computes multi-timeframe comparative alpha and cross-sectional universe percentile rankings.
"""

from typing import Any
import numpy as np
import pandas as pd


class RelativeStrengthEngine:
    """Computes Mansfield Relative Strength and comparative multi-period alpha."""

    @staticmethod
    def calculate_mansfield_rs(
        stock_close: pd.Series, benchmark_close: pd.Series, period: int = 50
    ) -> pd.Series:
        """
        Calculates Mansfield Relative Strength (RS).
        RS(t) = ((Stock_Close(t) / Benchmark_Close(t)) / SMA(Stock_Close/Benchmark_Close, 50) - 1.0) * 100.0
        """
        if len(stock_close) != len(benchmark_close) or stock_close.empty:
            # Reindex if length differs
            aligned_stock, aligned_bench = stock_close.align(benchmark_close, join="inner")
        else:
            aligned_stock, aligned_bench = stock_close, benchmark_close

        ratio = aligned_stock / aligned_bench.replace(0.0, np.nan)
        ratio_sma = ratio.rolling(window=period, min_periods=5).mean()
        mansfield_rs = ((ratio / ratio_sma.replace(0.0, np.nan)) - 1.0) * 100.0
        return mansfield_rs.fillna(0.0)

    @staticmethod
    def calculate_multi_period_alpha(
        stock_close: pd.Series, benchmark_close: pd.Series
    ) -> dict[str, float]:
        """
        Calculates outperformance (Alpha %) over 5D, 20D, 60D, and 120D windows.
        Alpha = Stock_Return% - Benchmark_Return%
        """
        if len(stock_close) < 5 or len(benchmark_close) < 5:
            return {"alpha_5d": 0.0, "alpha_20d": 0.0, "alpha_60d": 0.0, "alpha_120d": 0.0}

        windows = {"5d": 5, "20d": 20, "60d": 60, "120d": 120}
        alphas: dict[str, float] = {}

        for name, win in windows.items():
            if len(stock_close) > win and len(benchmark_close) > win:
                stock_ret = ((stock_close.iloc[-1] - stock_close.iloc[-win]) / stock_close.iloc[-win]) * 100.0
                bench_ret = ((benchmark_close.iloc[-1] - benchmark_close.iloc[-win]) / benchmark_close.iloc[-win]) * 100.0
                alphas[f"alpha_{name}"] = round(float(stock_ret - bench_ret), 2)
            else:
                alphas[f"alpha_{name}"] = 0.0

        return alphas

    @staticmethod
    def calculate_universe_percentile_ranks(
        rs_scores: dict[str, float]
    ) -> dict[str, float]:
        """
        Computes percentile ranking (0.0 to 100.0) across all symbols in the scanned universe.
        A score of 90.0 means outperforming 90% of all listed securities.
        """
        if not rs_scores:
            return {}

        symbols = list(rs_scores.keys())
        values = np.array([rs_scores[s] for s in symbols])

        # Rank values (scipy/numpy rankdata equivalent)
        temp = values.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(values))

        percentiles = (ranks / max(len(values) - 1, 1)) * 100.0
        return {sym: round(float(p), 1) for sym, p in zip(symbols, percentiles)}
