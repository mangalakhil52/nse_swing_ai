"""End-to-end NSE baseline-vs-P1 OOS experiment.

This is intentionally a manual/long-running research job. It downloads official
NSE daily bhavcopies, reconstructs the observed trading universe per day, loads
NIFTY 50 from the official NSE historical-index endpoint, and runs the strict
chronological OOS comparison.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from src.backtest.walk_forward import WalkForwardConfig
from src.data.nse_index_source import NSEHistoricalIndexSource
from src.research.nse_research_data import load_observed_universe
from src.research.strategy_comparison import ExperimentConfig, walk_forward_compare


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--alpha-threshold", type=float, default=0.10)
    parser.add_argument("--output", default="nse_oos_comparison.json")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    dataset = load_observed_universe(start, end)
    benchmark = NSEHistoricalIndexSource().fetch("NIFTY 50", start, end)
    result = walk_forward_compare(
        dataset.stock_dfs,
        benchmark,
        WalkForwardConfig(),
        ExperimentConfig(min_alpha_score=args.alpha_threshold),
    )
    result["dataset"] = dataset.diagnostics
    result["observed_universe_sessions"] = len(dataset.observed_universe_by_date)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
