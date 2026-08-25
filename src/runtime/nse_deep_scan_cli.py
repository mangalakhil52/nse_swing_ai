"""CLI for the deep NSE scan."""
from __future__ import annotations

import argparse
import json
from datetime import date

from src.runtime.nse_deep_scan import run


def main() -> None:
    parser = argparse.ArgumentParser(description="NSE Dynamic -> Technical -> CIO deep scan")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--lookback", type=int, default=260)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--technical-shortlist", type=int, default=50)
    args = parser.parse_args()

    summary, recommendations = run(
        as_of_date=args.as_of,
        lookback_calendar_days=args.lookback,
        max_workers=args.workers,
        technical_shortlist_size=args.technical_shortlist,
    )
    print(json.dumps({
        "summary": summary.__dict__,
        "recommendations": [r.model_dump(mode="json") if hasattr(r, "model_dump") else r.dict() for r in recommendations],
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
