from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    universe = int(summary.get("universe_count", 0))
    candidates = int(summary.get("candidate_count", 0))
    technical_success = int(summary.get("technical_success_count", 0))
    technical_failure = int(summary.get("technical_failure_count", 0))
    shortlist = int(summary.get("shortlist_count", 0))
    diagnostics = summary.get("historical_diagnostics", {})
    errors = list(diagnostics.get("errors", []))
    successful_days = int(diagnostics.get("successful_days", 0))
    attempted_days = int(diagnostics.get("attempted_days", 0))

    failures: list[str] = []
    if universe < 2000:
        failures.append(f"universe_count too low: {universe}")
    if candidates < 1:
        failures.append("candidate_count is zero")
    if technical_failure != 0:
        failures.append(f"technical_failure_count={technical_failure}")
    if technical_success != candidates:
        failures.append(f"technical_success_count={technical_success} != candidate_count={candidates}")
    if shortlist < 1:
        failures.append("technical shortlist is empty")
    if attempted_days and successful_days / attempted_days < 0.90:
        failures.append(f"historical coverage below 90%: {successful_days}/{attempted_days}")

    # Recent listings are intentionally not treated as verified IPOs until a
    # primary-market source is added. This prevents inferred listing dates from
    # silently becoming IPO facts downstream.
    for item in summary.get("ipo_shortlist", []):
        if item.get("listing_type") not in {"UNKNOWN", None}:
            continue
        if item.get("track") == "RECENT_IPO":
            failures.append(f"recent listing still mislabeled RECENT_IPO: {item.get('symbol')}")

    report = {
        "universe": universe,
        "candidates": candidates,
        "technical_success": technical_success,
        "technical_failure": technical_failure,
        "shortlist": shortlist,
        "historical": {
            "attempted_days": attempted_days,
            "successful_days": successful_days,
            "error_count": len(errors),
        },
        "failures": failures,
    }
    Path("scan_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "nse_dynamic_swing_scan.json"))
