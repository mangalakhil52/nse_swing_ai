from pathlib import Path

import pandas as pd
import pytest

from scripts.run_oos_validation import _load_csvs


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_csvs_requires_timestamp(tmp_path):
    _write_csv(tmp_path / "AAA.csv", [{"open": 1, "high": 2, "low": 0, "close": 1, "volume": 10}])
    with pytest.raises(ValueError, match="timestamp"):
        _load_csvs(tmp_path)


def test_load_csvs_fails_on_duplicate_timestamps(tmp_path):
    rows = [
        {"timestamp": "2026-01-01", "open": 1, "high": 2, "low": 0, "close": 1, "volume": 10},
        {"timestamp": "2026-01-01", "open": 1, "high": 2, "low": 0, "close": 1, "volume": 11},
    ]
    _write_csv(tmp_path / "AAA.csv", rows)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        _load_csvs(tmp_path)


def test_load_csvs_accepts_valid_symbol(tmp_path):
    _write_csv(
        tmp_path / "AAA.csv",
        [{"timestamp": "2026-01-01", "open": 1, "high": 2, "low": 0, "close": 1.5, "volume": 10}],
    )
    out = _load_csvs(tmp_path)
    assert list(out) == ["AAA"]
    assert out["AAA"]["timestamp"].iloc[0] == pd.Timestamp("2026-01-01")
