"""#14P production scan runtime entry point.

Configuration-driven and dry-run by default. External providers are injected so
runtime orchestration never embeds credentials or vendor-specific logic.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import date
from pathlib import Path


def build_runtime_config() -> dict:
    return {
        "universe_url": os.getenv("NSE_UNIVERSE_URL", ""),
        "market_data_url": os.getenv("MARKET_DATA_BASE_URL", ""),
        "output_dir": os.getenv("SCAN_OUTPUT_DIR", "data/runs"),
        "max_workers": int(os.getenv("SCAN_MAX_WORKERS", "8")),
        "mode": os.getenv("SCAN_MODE", "DRY_RUN").upper(),
    }


def persist_scan(result, output_dir: str) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"scan_{result.as_of_date.isoformat()}.json"
    payload = {
        "as_of_date": result.as_of_date.isoformat(),
        "items": [
            {"symbol": x.symbol, "stage": x.stage, "status": x.status,
             "error": x.error,
             "result": x.result if isinstance(x.result, (str, int, float, dict, list, type(None))) else str(x.result)}
            for x in result.items
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the configured NSE swing scanner")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--mode", choices=["DRY_RUN", "LIVE"], default=None)
    args = parser.parse_args(argv)
    config = build_runtime_config()
    mode = (args.mode or config["mode"]).upper()
    if mode not in {"DRY_RUN", "LIVE"}:
        raise ValueError("SCAN_MODE must be DRY_RUN or LIVE")
    if mode == "LIVE" and not config["universe_url"]:
        raise RuntimeError("NSE_UNIVERSE_URL is required for LIVE mode")
    # Wiring concrete providers is intentionally dependency-injected by the deployment layer.
    print(json.dumps({"status": "READY", "mode": mode, "as_of": args.as_of}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
