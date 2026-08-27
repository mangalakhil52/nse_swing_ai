"""Run baseline-vs-P1 OOS comparison on a prepared PIT historical dataset.

Input layout:
  --data-dir/RELIANCE.csv, --data-dir/TRENT.csv, ...
  --benchmark benchmark.csv

Each stock CSV must contain timestamp, open, high, low, close, volume.
The benchmark CSV must contain timestamp and close (OHLCV is also accepted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from src.backtest.walk_forward import WalkForwardConfig
from src.research.strategy_comparison import ExperimentConfig, walk_forward_compare


def load_csv_dir(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Historical data directory not found: {path}")
    result: dict[str, pd.DataFrame] = {}
    for csv in sorted(path.glob("*.csv")):
        frame = pd.read_csv(csv)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        if not frame.empty:
            result[csv.stem.upper()] = frame.sort_values("timestamp").reset_index(drop=True)
    if not result:
        raise ValueError("No valid stock CSVs were found")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--alpha-threshold", type=float, default=0.10)
    parser.add_argument("--output", default="oos_comparison.json")
    args = parser.parse_args()

    stocks = load_csv_dir(Path(args.data_dir))
    benchmark = pd.read_csv(args.benchmark)
    if "timestamp" not in benchmark.columns or "close" not in benchmark.columns:
        raise ValueError("Benchmark must contain timestamp and close")
    benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"], errors="coerce")
    benchmark = benchmark.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    result = walk_forward_compare(
        stocks,
        benchmark,
        WalkForwardConfig(
            train_days=args.train_days,
            validation_days=args.validation_days,
            test_days=args.test_days,
            step_days=args.step_days,
        ),
        ExperimentConfig(min_alpha_score=args.alpha_threshold),
    )
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
