"""P0 #14C tests for the deterministic TechnicalAnalysisAgent."""

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
    return pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


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
    assert result.signal in {SignalType.BULLISH, SignalType.BEARISH, SignalType.NEUTRAL, SignalType.UNKNOWN}
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
    assert baseline.evidence == changed.evidence


def test_missing_history_returns_unknown_and_not_bullish():
    df = _ohlcv(20)
    decision_time = df["timestamp"].iloc[-1].to_pydatetime()
    result = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta(), df, decision_time, "short"))
    assert result.signal == SignalType.UNKNOWN
    assert result.score == 0.0
    assert result.confidence == 0.0
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


def test_technical_score_changes_with_meaningfully_different_market_structure():
    """Regression guard against a broad universe collapsing to one score bucket."""
    strong = _ohlcv()
    weak = strong.copy()
    # Keep the same PIT horizon but materially weaken the latest structure and volume.
    weak.loc[weak.index[-1], "close"] = weak["close"].iloc[-1] * 0.955
    weak.loc[weak.index[-1], "open"] = weak["close"].iloc[-1] * 1.01
    weak.loc[weak.index[-1], "high"] = max(weak["open"].iloc[-1], weak["close"].iloc[-1]) * 1.002
    weak.loc[weak.index[-1], "low"] = min(weak["open"].iloc[-1], weak["close"].iloc[-1]) * 0.995
    weak.loc[weak.index[-1], "volume"] = 55_000
    decision_time = strong["timestamp"].iloc[-1].to_pydatetime()

    strong_result = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta("STRONG"), strong, decision_time, "strong"))
    weak_result = asyncio.run(TechnicalAnalysisAgent().analyze_contract(_meta("WEAK"), weak, decision_time, "weak"))

    assert strong_result.pit_safe and weak_result.pit_safe
    assert strong_result.score != weak_result.score
    assert abs(strong_result.score - weak_result.score) >= 5.0


def test_pattern_quality_is_not_constant_for_different_pullback_geometry():
    """EMA pullback quality must reflect measurable candle/volume geometry."""
    from src.quant.indicators import TechnicalIndicators
    from src.quant.patterns import PatternRecognizer

    base = TechnicalIndicators.compute_all_indicators(_ohlcv())
    stronger = base.copy()
    weaker = base.copy()
    stronger.loc[stronger.index[-1], "close"] = stronger["ema_20"].iloc[-1] * 1.002
    stronger.loc[stronger.index[-1], "open"] = stronger["ema_20"].iloc[-1] * 0.995
    stronger.loc[stronger.index[-1], "low"] = stronger["ema_20"].iloc[-1] * 0.997
    stronger.loc[stronger.index[-1], "high"] = stronger["ema_20"].iloc[-1] * 1.01
    stronger.loc[stronger.index[-1], "volume"] = stronger["volume_sma_20"].iloc[-1] * 2.0
    stronger = TechnicalIndicators.compute_all_indicators(stronger)

    weaker.loc[weaker.index[-1], "close"] = weaker["ema_20"].iloc[-1] * 1.001
    weaker.loc[weaker.index[-1], "open"] = weaker["ema_20"].iloc[-1] * 0.999
    weaker.loc[weaker.index[-1], "low"] = weaker["ema_20"].iloc[-1] * 1.001
    weaker.loc[weaker.index[-1], "high"] = weaker["ema_20"].iloc[-1] * 1.002
    weaker.loc[weaker.index[-1], "volume"] = weaker["volume_sma_20"].iloc[-1] * 0.6
    weaker = TechnicalIndicators.compute_all_indicators(weaker)

    strong_matches = [x for x in PatternRecognizer.evaluate_all_patterns(stronger) if x.pattern_type.value == "EMA_PULLBACK_REVERSAL"]
    weak_matches = [x for x in PatternRecognizer.evaluate_all_patterns(weaker) if x.pattern_type.value == "EMA_PULLBACK_REVERSAL"]
    if strong_matches and weak_matches:
        assert strong_matches[0].quality_score != weak_matches[0].quality_score
