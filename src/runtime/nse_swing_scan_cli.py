"""CLI for the dynamic whole-NSE swing scanner."""
from __future__ import annotations

import argparse
import json
from datetime import date

from src.runtime.nse_swing_scan import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scan the complete NSE equity universe dynamically")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--lookback", type=int, default=260)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=50)
    parser.add_argument("--min-price", type=float, default=None)
    parser.add_argument("--min-adtv-crores", type=float, default=None)
    args = parser.parse_args(argv)

    summary, shortlist = run(
        as_of_date=date.fromisoformat(args.as_of),
        lookback_calendar_days=args.lookback,
        max_workers=args.workers,
        shortlist_size=args.shortlist,
        min_price=args.min_price,
        min_adtv_crores=args.min_adtv_crores,
    )

    print(json.dumps({
        "summary": asdict(summary),
        "shortlist": [asdict(row) for row in shortlist],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    from dataclasses import asdict
    raise SystemExit(main())
