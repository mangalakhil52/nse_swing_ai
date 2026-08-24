"""#14N market data adapter tests."""
from datetime import date
from unittest.mock import Mock, patch
import pandas as pd
import pytest
from src.data.market_data_adapter import MarketDataAdapter


def _df():
    return pd.DataFrame({
        "timestamp": ["2026-06-29", "2026-06-30", "2026-07-01"],
        "open": [100, 101, 102], "high": [102, 103, 104], "low": [99, 100, 101],
        "close": [101, 102, 103], "volume": [1000, 1100, 1200],
    })


def test_filters_future_rows_and_sorts():
    out = MarketDataAdapter(lambda _: _df()).fetch("TRENT", date(2026, 6, 30))
    assert out.frame["timestamp"].dt.date.tolist() == [date(2026,6,29), date(2026,6,30)]


def test_missing_columns_fail_closed():
    bad = pd.DataFrame({"timestamp": ["2026-06-30"], "close": [100]})
    with pytest.raises(RuntimeError, match="required columns"):
        MarketDataAdapter(lambda _: bad, retries=1).fetch("TRENT", date(2026,6,30))


def test_empty_after_pit_filter_fails_closed():
    future = _df().iloc[[2]]
    with pytest.raises(RuntimeError, match="point-in-time"):
        MarketDataAdapter(lambda _: future, retries=1).fetch("TRENT", date(2026,6,30))


def test_transient_fetch_retries_with_backoff():
    fetcher = Mock(side_effect=[OSError("temporary"), _df()])
    with patch("src.data.market_data_adapter.time.sleep") as sleep:
        out = MarketDataAdapter(fetcher, retries=2).fetch("TRENT", date(2026,6,30))
    assert len(out.frame) == 2
    sleep.assert_called_once_with(1)


def test_duplicate_timestamps_keep_latest_row():
    frame = _df()
    frame = pd.concat([frame, frame.iloc[[1]].assign(close=999)], ignore_index=True)
    out = MarketDataAdapter(lambda _: frame).fetch("TRENT", date(2026,6,30))
    assert out.frame.loc[out.frame["timestamp"].dt.date == date(2026,6,30), "close"].iloc[0] == 999
