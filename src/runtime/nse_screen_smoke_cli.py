"""CLI for a controlled real-NSE Candidate Discovery smoke run."""
from __future__ import annotations
import argparse
import json
from collections import Counter
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

    rejection_counts = Counter()
    for result in results:
        for reason in result.reasons:
            rejection_counts[reason] += 1
        if not result.eligible and not result.reasons:
            rejection_counts["NO_REJECTION_REASON"] += 1

    result_rows = [
        {
            "symbol": str(r.symbol),
            "eligible": r.eligible,
            "score": r.discovery_score,
            "passed_filters": r.passed_filters,
            "failed_filters": r.failed_filters,
            "reasons": r.reasons,
            "pit_safe": r.pit_safe,
            "data_quality": r.data_quality.overall_status.value if r.data_quality else None,
        }
        for r in results
    ]

    print(json.dumps({
        "summary": summary.__dict__,
        "rejection_counts": dict(rejection_counts.most_common()),
        "results": result_rows,
        "candidates": [r for r in result_rows if r["eligible"]],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
