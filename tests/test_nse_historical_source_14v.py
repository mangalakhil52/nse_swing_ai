"""#14V historical NSE OHLCV tests."""
from datetime import date
import io
import zipfile
import pandas as pd
from src.data.nse_historical_source import NSEHistoricalOHLCVSource


def _zip(day, symbol="TRENT", close=100):
    csv = pd.DataFrame({"TckrSymb":[symbol],"TradDt":[day.isoformat()],"OpnPric":[99],"HghPric":[101],"LwPric":[98],"ClsPric":[close],"TtlTradgVol":[10000]}).to_csv(index=False).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("bhav.csv", csv)
    return buf.getvalue()


def test_historical_source_builds_multi_day_history_and_caches():
    calls = []
    def fetch(day):
        calls.append(day)
        return _zip(day, close=100 + len(calls))
    source = NSEHistoricalOHLCVSource(date(2026, 6, 30), lookback_calendar_days=2, fetcher=fetch)
    out = source.fetch("TRENT")
    assert len(out) == 2
    assert out["close"].tolist() == [101, 102]
    assert len(calls) == 2
    source.fetch("TRENT")
    assert len(calls) == 2


def test_future_rows_are_excluded():
    def fetch(day):
        return _zip(date(2026, 7, 1), close=999)
    source = NSEHistoricalOHLCVSource(date(2026, 6, 30), lookback_calendar_days=2, fetcher=fetch)
    try:
        source.fetch("TRENT")
    except ValueError:
        pass
    else:
        raise AssertionError("future-only data must fail closed")
