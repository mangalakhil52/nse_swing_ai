from datetime import datetime, date

from config.market_hours import IST, resolve_scan_date


def test_live_scan_uses_previous_completed_session_before_eod():
    now = datetime(2026, 9, 7, 14, 0, tzinfo=IST)
    assert resolve_scan_date(now=now) == date(2026, 9, 3)


def test_live_scan_uses_today_after_eod_cutoff():
    now = datetime(2026, 9, 7, 15, 45, tzinfo=IST)
    assert resolve_scan_date(now=now) == date(2026, 9, 7)


def test_live_scan_walks_back_from_weekend():
    now = datetime(2026, 9, 6, 12, 0, tzinfo=IST)
    assert resolve_scan_date(now=now) == date(2026, 9, 3)


def test_explicit_scan_date_is_always_honored():
    now = datetime(2026, 9, 7, 14, 0, tzinfo=IST)
    requested = date(2026, 9, 1)
    assert resolve_scan_date(requested_date=requested, now=now) == requested
