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


def test_bulk_fetch_downloads_each_day_once_for_many_symbols():
    calls = []
    def fetch(day):
        calls.append(day)
        first = _zip(day, "TRENT", 101)
        second = _zip(day, "INFY", 201)
        with zipfile.ZipFile(io.BytesIO(first)) as z1, zipfile.ZipFile(io.BytesIO(second)) as z2:
            frames = []
            for z in (z1, z2):
                frames.append(pd.read_csv(z.open("bhav.csv")))
        csv = pd.concat(frames, ignore_index=True).to_csv(index=False).encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z: z.writestr("bhav.csv", csv)
        return buf.getvalue()
    source = NSEHistoricalOHLCVSource(date(2026, 6, 30), lookback_calendar_days=2, fetcher=fetch)
    out = source.fetch_many(["TRENT", "INFY"])
    assert set(out) == {"TRENT", "INFY"}
    assert len(calls) == 2
    assert source.diagnostics.successful_days == 2
    assert source.diagnostics.matching_days == 2


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
