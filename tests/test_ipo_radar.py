from datetime import date, timedelta

import pandas as pd

from src.runtime.ipo_radar import RecentIPORadar


def _df(start=date(2026, 6, 1), n=25, first=100.0):
    rows = []
    for i in range(n):
        d = start + timedelta(days=i)
        rows.append({
            "timestamp": d,
            "open": first + i,
            "high": first + i + 2,
            "low": first + i - 1,
            "close": first + i,
            "volume": 500_000,
        })
    return pd.DataFrame(rows)


def test_recent_listing_is_detected_without_long_history():
    radar = RecentIPORadar(date(2026, 6, 30), max_age_days=180, min_bars=10, min_turnover_crores=0.5)
    row = radar.evaluate("NEWCO", _df(n=20))
    assert row is not None
    assert row.symbol == "NEWCO"
    assert row.listing_age_days < 180
    assert row.bars == 20
    assert row.track == "RECENT_IPO"


def test_old_first_bar_is_not_treated_as_recent_ipo():
    radar = RecentIPORadar(date(2026, 8, 24), max_age_days=180, min_bars=10)
    assert radar.evaluate("OLDCO", _df(start=date(2025, 1, 1), n=25)) is None


def test_short_history_can_be_radar_candidate_even_when_normal_screen_would_reject():
    radar = RecentIPORadar(date(2026, 8, 24), max_age_days=180, min_bars=10, min_turnover_crores=0.5)
    row = radar.evaluate("IPOCO", _df(start=date(2026, 7, 20), n=15))
    assert row is not None
    assert row.bars < 50


def test_data_gap_does_not_create_false_ipo_signal():
    df = _df(start=date(2026, 6, 1), n=20)
    df.loc[10:, "timestamp"] = pd.to_datetime(df.loc[10:, "timestamp"]) + pd.Timedelta(days=20)
    radar = RecentIPORadar(date(2026, 8, 24), max_age_days=180, min_bars=10, min_turnover_crores=0.5)
    assert radar.evaluate("GAPCO", df) is None
