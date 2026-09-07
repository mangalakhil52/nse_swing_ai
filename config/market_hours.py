"""
NSE Market Hours and Calendar Utility Module.
Handles Indian Standard Time (IST) sessions, pre-market, live market, post-market, and holiday schedules.
"""

from datetime import datetime, time, date, timedelta

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except ImportError:
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

# NSE Standard Equity Market Hours (IST)
PRE_MARKET_OPEN = time(9, 0)
PRE_MARKET_CLOSE = time(9, 8)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
POST_MARKET_CLOSE = time(16, 0)
EOD_DATA_AVAILABLE = time(15, 45)


# Standard 2026 NSE Holidays Calendar (Key Trading Holidays)
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 6),    # Maha Shivratri
    date(2026, 3, 25),   # Holi
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 6, 17),   # Bakri Id
    date(2026, 8, 15),   # Independence Day
    date(2026, 9, 4),    # Janmashtami
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 8),   # Diwali Laxmi Pujan (Muhurat trading in evening)
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Gurunanak Jayanti
    date(2026, 12, 25),  # Christmas
}


def get_current_ist_datetime() -> datetime:
    """Returns current datetime in Indian Standard Time (Asia/Kolkata)."""
    return datetime.now(IST)


def is_trading_day(check_date: date | None = None) -> bool:
    """Checks if a given date is an official NSE trading day (Mon-Fri and not a holiday)."""
    if check_date is None:
        check_date = get_current_ist_datetime().date()

    if check_date.weekday() >= 5:
        return False
    if check_date in NSE_HOLIDAYS_2026:
        return False
    return True


def get_latest_trading_day(check_date: date | None = None) -> date:
    """
    Returns the most recent completed NSE trading session.

    For today's live execution, today's session is considered complete only after
    EOD_DATA_AVAILABLE (15:45 IST). Before that cutoff, the function returns the
    preceding completed trading session. For weekends/holidays it walks backward.
    """
    if check_date is None:
        check_date = get_current_ist_datetime().date()

    curr = check_date
    if check_date == get_current_ist_datetime().date() and not is_eod_scan_ready():
        curr -= timedelta(days=1)

    while not is_trading_day(curr):
        curr -= timedelta(days=1)

    return curr


def resolve_scan_date(requested_date: date | None = None, now: datetime | None = None) -> date:
    """
    Resolve the date used by the daily scanner.

    Explicit ``requested_date`` is always honored for historical/manual runs.
    Without it, use the current IST session: today's date after the EOD data
    cutoff, otherwise the previous completed trading day. This makes the live
    scanner deterministic and prevents use of an incomplete current-day candle.
    """
    if requested_date is not None:
        return requested_date

    current = now.astimezone(IST) if now is not None else get_current_ist_datetime()
    if is_trading_day(current.date()) and current.time() >= EOD_DATA_AVAILABLE:
        return current.date()

    curr = current.date() - timedelta(days=1) if is_trading_day(current.date()) else current.date()
    while not is_trading_day(curr):
        curr -= timedelta(days=1)
    return curr


def is_market_open(dt: datetime | None = None) -> bool:
    """Checks if NSE live trading session is currently active (09:15 to 15:30 IST on trading days)."""
    if dt is None:
        dt = get_current_ist_datetime()

    if not is_trading_day(dt.date()):
        return False

    current_time = dt.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_eod_scan_ready(dt: datetime | None = None) -> bool:
    """Checks if confirmed official post-market EOD Bhavcopy data is ready for analysis (>= 15:45 IST)."""
    if dt is None:
        dt = get_current_ist_datetime()

    if not is_trading_day(dt.date()):
        return True

    return dt.time() >= EOD_DATA_AVAILABLE


def get_next_trading_sessions(start_date: date, n: int) -> list[date]:
    """Returns the next n valid NSE trading session dates after start_date."""
    sessions = []
    curr = start_date + timedelta(days=1)
    while len(sessions) < n:
        if is_trading_day(curr):
            sessions.append(curr)
        curr += timedelta(days=1)
    return sessions


def get_previous_trading_sessions(start_date: date, n: int) -> list[date]:
    """Returns the previous n valid NSE trading session dates before start_date."""
    sessions = []
    curr = start_date - timedelta(days=1)
    while len(sessions) < n:
        if is_trading_day(curr):
            sessions.append(curr)
        curr -= timedelta(days=1)
    return sessions


class MarketCalendar:
    """Class wrapper for calendar and session checks."""

    get_current_ist_datetime = staticmethod(get_current_ist_datetime)
    is_trading_day = staticmethod(is_trading_day)
    get_latest_trading_day = staticmethod(get_latest_trading_day)
    resolve_scan_date = staticmethod(resolve_scan_date)
    get_next_trading_sessions = staticmethod(get_next_trading_sessions)
    get_previous_trading_sessions = staticmethod(get_previous_trading_sessions)
    is_market_open = staticmethod(is_market_open)
    is_eod_scan_ready = staticmethod(is_eod_scan_ready)

