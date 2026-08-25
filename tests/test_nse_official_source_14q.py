"""Tests for official NSE public-file sources."""
from datetime import date
from io import BytesIO
from unittest.mock import Mock, patch
import zipfile
import pandas as pd
from src.data.nse_official_source import NSEOfficialUniverseSource, NSEOfficialBhavcopySource


def test_official_universe_keeps_equity_series_only():
    csv = b"SYMBOL,NAME OF COMPANY,SERIES\nTRENT,Trent Ltd,EQ\nINFY,Infosys,BE\nTEST,Test,SM\n"
    response = Mock(content=csv)
    response.raise_for_status.return_value = None
    with patch("src.data.nse_official_source.requests.Session") as session_cls:
        session_cls.return_value.__enter__.return_value.get.return_value = response
        rows = NSEOfficialUniverseSource().fetch()
    assert [r["symbol"] for r in rows] == ["TRENT", "INFY"]


def test_official_bhavcopy_parses_udiff_and_caches_daily_file():
    frame = pd.DataFrame({
        "TckrSymb": ["TRENT", "INFY"], "TradDt": ["2026-08-20", "2026-08-20"],
        "OpnPric": [100, 200], "HghPric": [110, 210], "LwPric": [95, 195],
        "ClsPric": [108, 205], "TtlTradgVol": [10000, 20000],
    })
    raw = frame.to_csv(index=False).encode()
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BhavCopy.csv", raw)
    response = Mock(content=buf.getvalue())
    response.raise_for_status.return_value = None
    source = NSEOfficialBhavcopySource(date(2026, 8, 20))
    with patch("src.data.nse_official_source.requests.Session") as session_cls:
        session_cls.return_value.__enter__.return_value.get.return_value = response
        a = source.fetch("TRENT")
        b = source.fetch("INFY")
    assert a.iloc[0]["close"] == 108
    assert b.iloc[0]["volume"] == 20000
    session_cls.return_value.__enter__.return_value.get.assert_called_once()
