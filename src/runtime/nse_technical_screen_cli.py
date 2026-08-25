"""CLI for Stage-2 NSE Candidate Discovery -> Technical Analysis."""
from __future__ import annotations
import argparse
import json
from datetime import date
from src.runtime.nse_technical_screen import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=140)
    args = parser.parse_args(argv)
    summary, results = run(date.fromisoformat(args.as_of), args.lookback, args.workers, args.limit)
    print(json.dumps({
        "summary": asdict(summary),
        "results": results,
        "bullish_candidates": [row for row in results if row["signal"] == "BULLISH"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    from dataclasses import asdict
    raise SystemExit(main())
