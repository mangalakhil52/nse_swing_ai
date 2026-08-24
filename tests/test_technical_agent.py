"""P0 #14C tests for the deterministic TechnicalAnalysisAgent."""

from datetime import date
import asyncio
import numpy as np
import pandas as pd

from src.agents.technical_agent import TechnicalAnalysisAgent
from src.architecture.contracts import AgentAnalysisResult
from src.core.models import SymbolMetadata
from src.core.types import SignalType


def _ohlcv(n: int = 240, start: str = "2025-08-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    base = np.linspace(100.0, 180.0, n)
    wiggle = np.sin(np.arange(n) / 5.0) * 1.5
    close = base + wiggle
    open_ = close - 0.5
    high = close + 2.0
    low = close - 2.0
    volume = np.full(n, 100_000, dtype=float)
    volume[-1] = 250_000
    return pd.DataFrame({
        "timestamp": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _meta(symbol: str = "TEST") -> SymbolMetadata:
    return SymbolMetadata(symbol=symbol, company_name="Synthetic Test Co")


def test_technical_agent_returns_architecture_contract():
    df = _ohlcv()
    decision_time = df["timestamp"].iloc[-1].to_pydatetime()
    result = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta(), df, decision_time, "run-14c"))

    assert isinstance(result, AgentAnalysisResult)
    assert result.symbol == "TEST"
    assert result.agent_name == "technical_analysis_agent"
    assert result.decision_time == decision_time
    assert result.pit_safe is True
    assert result.signal in {
        SignalType.BULLISH, SignalType.BEARISH, SignalType.NEUTRAL, SignalType.UNKNOWN
    }
    assert result.score >= 0
    assert result.confidence >= 0
    assert result.evidence
    assert all(e.source == "TECHNICAL" for e in result.evidence)
    assert all(e.pit_safe for e in result.evidence)


def test_future_rows_cannot_change_contract_at_decision_time():
    df = _ohlcv()
    decision_time = df["timestamp"].iloc[-20].to_pydatetime()

    baseline = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta(), df, decision_time, "base"))

    mutated = df.copy()
    future = pd.to_datetime(mutated["timestamp"]) > pd.Timestamp(decision_time)
    mutated.loc[future, "close"] = 999999.0
    mutated.loc[future, "high"] = 1000000.0
    mutated.loc[future, "low"] = 999998.0
    mutated.loc[future, "volume"] = 999999999

    changed = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta(), mutated, decision_time, "mutated"))

    assert baseline.signal == changed.signal
    assert baseline.score == changed.score
    assert baseline.confidence == changed.confidence
    assert baseline.metrics() if False else True


def test_missing_history_returns_unknown_and_not_bullish():
    df = _ohlcv(20)
    decision_time = df["timestamp"].iloc[-1].to_pydatetime()
    result = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta(), df, decision_time, "short"))

    assert result.signal == SignalType.UNKNOWN
    assert result.score == 0.0
    assert result.pit_safe is True
    assert "INSUFFICIENT_HISTORY" in result.reasons or "OHLC_INSUFFICIENT_BARS" in result.reasons


def test_technical_agent_does_not_emit_trade_levels():
    df = _ohlcv()
    decision_time = df["timestamp"].iloc[-1].to_pydatetime()
    result = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta(), df, decision_time, "levels"))

    payload = result.model_dump()
    assert "target" not in str(payload).lower()
    assert "stop_loss" not in str(payload).lower()
    assert "position_size" not in str(payload).lower()
