#!/usr/bin/env python3
"""Strict walk-forward / out-of-sample validation entry point.

Input is a directory containing one CSV per symbol. CSVs must contain either a
`timestamp` column or a datetime index plus OHLCV columns required by the
production portfolio engine. No current/live universe fallback is permitted.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardValidator


def _load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"OOS data directory does not exist: {data_dir}")
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"OOS data directory contains no CSV files: {data_dir}")

    result: dict[str, pd.DataFrame] = {}
    for path in files:
        symbol = path.stem.upper()
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"{path}: required timestamp column is missing")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(c.lower() for c in df.columns)
        if missing:
            raise ValueError(f"{path}: missing OHLCV columns: {sorted(missing)}")
        result[symbol] = df
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict production walk-forward OOS validation")
    parser.add_argument("--data-dir", required=True, help="Directory of point-in-time historical symbol CSVs")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--warmup-days", type=int, default=200)
    parser.add_argument("--output", default="artifacts/oos_validation.json")
    args = parser.parse_args()

    stock_dfs = _load_csvs(Path(args.data_dir))
    config = WalkForwardConfig(
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        warmup_days=args.warmup_days,
    )
    report = WalkForwardValidator.run_walk_forward(stock_dfs, config=config)

    payload = {
        "status": report.status,
        "config": vars(report.config),
        "windows": [vars(w) for w in report.windows],
        "robustness_metrics": report.robustness_metrics,
        "leakage_checks": report.leakage_checks,
        "calibration_performed": report.calibration_performed,
        "calibration_method": report.calibration_method,
        "frozen_configuration_hash": report.frozen_configuration_hash,
        "candidate_outcome_labels_count": report.candidate_outcome_labels_count,
        "eligible_outcome_labels_count": report.eligible_outcome_labels_count,
        "consumed_outcome_labels_count": report.consumed_outcome_labels_count,
        "oos_trade_count": len(report.oos_completed_trades),
        "oos_snapshot_count": len(report.oos_equity_snapshots),
        "rejection_reason": report.rejection_reason,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(json.dumps(payload, indent=2, default=str))
    if report.status != "OK":
        return 2
    if not all(report.leakage_checks.values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
