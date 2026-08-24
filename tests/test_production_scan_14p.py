"""#14P production runtime tests."""
import json
from datetime import date
from types import SimpleNamespace
from src.runtime.production_scan import build_runtime_config, main, persist_scan


def test_runtime_defaults_to_dry_run(monkeypatch):
    monkeypatch.delenv("SCAN_MODE", raising=False)
    assert build_runtime_config()["mode"] == "DRY_RUN"


def test_live_mode_requires_universe_url(monkeypatch):
    monkeypatch.delenv("NSE_UNIVERSE_URL", raising=False)
    assert main(["--mode", "DRY_RUN"]) == 0
    try:
        main(["--mode", "LIVE"])
    except RuntimeError as exc:
        assert "NSE_UNIVERSE_URL" in str(exc)
    else:
        raise AssertionError("LIVE mode must fail closed without universe source")


def test_scan_persistence_is_deterministic(tmp_path):
    result = SimpleNamespace(as_of_date=date(2026,6,30), items=(
        SimpleNamespace(symbol="TRENT", stage="DECISION", status="SUCCESS", error=None, result={"action":"BUY"}),
    ))
    path = persist_scan(result, str(tmp_path))
    payload = json.loads(path.read_text())
    assert payload["as_of_date"] == "2026-06-30"
    assert payload["items"][0]["symbol"] == "TRENT"
