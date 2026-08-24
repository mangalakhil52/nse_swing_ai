"""#14S production intelligence bridge tests."""
from datetime import date
import pandas as pd
from unittest.mock import Mock
from src.runtime.intelligence_pipeline import ProductionIntelligencePipeline


def _ohlcv():
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-04-01", periods=60),
        "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 100000,
    })


def test_discovery_uses_existing_engine():
    pipeline = ProductionIntelligencePipeline()
    result = pipeline.discover("TRENT", _ohlcv(), date(2026, 5, 30))
    assert result.symbol == "TRENT"
    assert result.pit_safe is True


def test_decision_delegates_to_existing_cio():
    cio = Mock()
    cio.analyze_candidate.return_value = ("decision", {"score": 1.0})
    pipeline = ProductionIntelligencePipeline(cio=cio)
    out = pipeline.decide("TRENT", _ohlcv(), date(2026, 5, 30))
    assert out == ("decision", {"score": 1.0})
    cio.analyze_candidate.assert_called_once()


def test_no_synthetic_downstream_context_is_created():
    cio = Mock()
    cio.analyze_candidate.return_value = (None, {})
    pipeline = ProductionIntelligencePipeline(cio=cio)
    pipeline.decide("TRENT", _ohlcv(), date(2026, 5, 30))
    kwargs = cio.analyze_candidate.call_args.kwargs
    assert kwargs["context"] == {}
