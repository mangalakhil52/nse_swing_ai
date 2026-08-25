"""CLI for a controlled real-NSE Candidate Discovery smoke run."""
from __future__ import annotations
import argparse
import json
from datetime import date
from src.runtime.nse_screen_smoke import run


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lookback", type=int, default=140)
    args = p.parse_args(argv)
    summary, results = run(date.fromisoformat(args.as_of), args.lookback, args.workers, args.limit)
    print(json.dumps({"summary": summary.__dict__, "candidates": [
        {"symbol": r.symbol, "eligible": r.eligible, "score": r.discovery_score,
         "reasons": r.blocking_reasons, "pit_safe": r.pit_safe}
        for r in results if r.eligible
    ]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
