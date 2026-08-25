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


def _legacy_zip(day, symbol="TRENT", close=100):
    csv = pd.DataFrame({"SYMBOL":[symbol],"SERIES":["EQ"],"TIMESTAMP":[day.strftime("%d-%b-%Y")],"OPEN":[99],"HIGH":[101],"LOW":[98],"CLOSE":[close],"TOTTRDQTY":[10000]}).to_csv(index=False).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("cm_bhav.csv", csv)
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


def test_pre_udiff_date_uses_legacy_archive_path_and_parser():
    source = NSEHistoricalOHLCVSource(date(2024, 7, 5))
    payload = _legacy_zip(date(2024, 7, 5), close=123)
    captured = {}
    source._download_url = lambda url, day: captured.setdefault("url", url) or payload
    out = source._download(date(2024, 7, 5))
    parsed = source._parse(out, date(2024, 7, 5))
    assert "/content/historical/EQUITIES/2024/JUL/" in captured["url"]
    assert "cm05JUL2024bhav.csv.zip" in captured["url"]
    assert parsed["close"].tolist() == [123]


def test_udiff_cutover_date_uses_modern_archive_path():
    source = NSEHistoricalOHLCVSource(date(2024, 7, 8))
    captured = {}
    source._download_report_api = lambda day: (_ for _ in ()).throw(FileNotFoundError(str(day)))
    source._download_url = lambda url, day: captured.setdefault("url", url) or _zip(day, close=123)
    source._download(date(2024, 7, 8))
    assert "/content/cm/BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip" in captured["url"]
