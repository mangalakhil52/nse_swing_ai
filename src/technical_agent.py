"""
P0 #14C — Deterministic Technical Analysis Agent.

Consumes a PIT-safe OHLCV series for one candidate and emits only
technical evidence. It does not perform fundamentals, news, regime,
conviction, risk, trade construction, or execution.

The implementation deliberately reuses the strategy's existing technical
ideas: EMA trend alignment, RSI, volume expansion, 20-session breakout,
20-session momentum, ATR, and the five established pattern families.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.core.models import AgentOutput, EvidenceItem
from src.data.data_quality import DataQualityGate, DataQualityStatus
from src.data.point_in_time import PointInTimeFilter


class TechnicalAgentConfig(BaseModel):
    min_history_bars: int = Field(default=60, ge=30)
    ema_fast: int = Field(default=21, ge=2)
    ema_slow: int = Field(default=50, ge=5)
    rsi_period: int = Field(default=14, ge=2)
    breakout_lookback: int = Field(default=20, ge=5)
    volume_lookback: int = Field(default=20, ge=5)
    breakout_volume_multiple: float = Field(default=1.5, gt=0)


class TechnicalAnalysisSnapshot(BaseModel):
    symbol: str
    decision_time: datetime | date
    signal: str
    score: float = Field(ge=0.0, le=100.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks_identified: list[str] = Field(default_factory=list)
    pit_safe: bool
    reasons: list[str] = Field(default_factory=list)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize repository OHLCV columns without changing the source data."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    rename = {}
    for col in out.columns:
        key = str(col).strip().lower()
        rename[col] = key
    out = out.rename(columns=rename)
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - previous).abs(),
         (df["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _detect_pattern(df: pd.DataFrame) -> tuple[str | None, dict[str, Any]]:
    """Small deterministic pattern adapter matching the existing strategy vocabulary."""
    if len(df) < 30:
        return None, {}

    today = df.iloc[-1]
    previous = df.iloc[-21:-1]
    vol_avg = float(previous["volume"].mean()) if not previous.empty else 0.0
    vol_ratio = float(today["volume"] / vol_avg) if vol_avg > 0 else 0.0

    # Range breakout: established strategy condition, expressed on normalized columns.
    if len(df) >= 25:
        rh = float(previous["high"].max())
        rl = float(previous["low"].min())
        width = (rh - rl) / rl if rl > 0 else float("inf")
        if today["close"] > rh and width <= 0.12 and vol_ratio >= 2.0:
            return "range_breakout", {"range_high": rh, "range_low": rl, "range_width_pct": width * 100, "vol_ratio": vol_ratio}

    # Inside-bar breakout.
    if len(df) >= 30:
        mother = df.iloc[-3]
        inside = df.iloc[-2]
        ema21 = df["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        if (inside["high"] <= mother["high"] and inside["low"] >= mother["low"]
                and today["close"] > mother["high"] and today["close"] >= ema21
                and vol_ratio >= 1.5):
            tightness = 1 - (inside["high"] - inside["low"]) / mother["high"]
            return "inside_bar", {"mother_high": float(mother["high"]), "mother_low": float(mother["low"]), "tightness": float(tightness), "vol_ratio": vol_ratio}

    # EMA pullback/bounce.
    if len(df) >= 60:
        ema21 = df["close"].ewm(span=21, adjust=False).mean()
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        rsi = _rsi(df["close"], 14).iloc[-1]
        touched = today["low"] <= ema21.iloc[-1] * 1.01 and today["low"] >= ema21.iloc[-1] * 0.97
        if (today["close"] > ema50.iloc[-1] and ema50.iloc[-1] > ema50.iloc[-6]
                and touched and today["close"] > ema21.iloc[-1]
                and today["close"] > today["open"] and 35 <= rsi <= 65
                and vol_ratio >= 0.8):
            return "ema_pullback", {"ema21": float(ema21.iloc[-1]), "ema50": float(ema50.iloc[-1]), "rsi": float(rsi), "vol_ratio": vol_ratio}

    return None, {"vol_ratio": vol_ratio}


class TechnicalAgent:
    """Deterministic technical specialist for the multi-agent pipeline."""

    AGENT_NAME = "TechnicalAgent"

    @classmethod
    def analyze(
        cls,
        symbol: str,
        market_data: pd.DataFrame,
        decision_time: datetime | date,
        config: TechnicalAgentConfig | None = None,
        run_id: str = "",
    ) -> AgentOutput:
        cfg = config or TechnicalAgentConfig()
        symbol = symbol.upper().strip()

        # Never allow the specialist to consume future OHLCV.
        pit_df = PointInTimeFilter.filter_market_data(market_data, decision_time)
        if pit_df is None or pit_df.empty:
            return cls._result(symbol, run_id, decision_time, "UNKNOWN", 0.0, None,
                               {}, [], ["NO_OHLCV_DATA"], True, True, "NO_OHLCV_DATA")

        df = _normalize_ohlcv(pit_df)
        if df.empty:
            return cls._result(symbol, run_id, decision_time, "UNKNOWN", 0.0, None,
                               {}, [], ["INVALID_OHLCV_SCHEMA"], True, False, "INVALID_OHLCV_SCHEMA")

        dq = DataQualityGate.evaluate_ohlcv(
            df, symbol, decision_time, min_required_bars=cfg.min_history_bars
        )
        if dq.status in (DataQualityStatus.INVALID, DataQualityStatus.PIT_VIOLATION):
            return cls._result(symbol, run_id, decision_time, "UNKNOWN", 0.0, 0.0,
                               {"data_quality_status": dq.status.value}, [],
                               list(dq.reasons), True, False, "TECHNICAL_DATA_QUALITY_FAILURE")

        if len(df) < cfg.min_history_bars:
            return cls._result(symbol, run_id, decision_time, "UNKNOWN", 0.0, 0.0,
                               {"bars": len(df)}, [], ["INSUFFICIENT_HISTORY"], True, False, "INSUFFICIENT_HISTORY")

        close = df["close"]
        ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        rsi = _rsi(close, cfg.rsi_period)
        atr = _atr(df, 14)
        volume_avg = df["volume"].iloc[-cfg.volume_lookback-1:-1].mean()
        volume_ratio = float(df["volume"].iloc[-1] / volume_avg) if volume_avg > 0 else 0.0
        prior_high = float(df["high"].iloc[-cfg.breakout_lookback-1:-1].max())
        last_close = float(close.iloc[-1])
        momentum_20 = float((last_close / close.iloc[-21]) - 1.0)
        pattern, pattern_data = _detect_pattern(df)

        bullish = 0
        bearish = 0
        evidence: list[EvidenceItem] = []

        def add(metric: str, value: Any, unit: str, direction: str, weight: int) -> None:
            nonlocal bullish, bearish
            evidence.append(EvidenceItem(
                metric_name=metric,
                observed_value=value,
                unit=unit,
                source="TechnicalAgent",
                timestamp=decision_time,
                verification_status="VERIFIED",
            ))
            if direction == "BULLISH":
                bullish += weight
            elif direction == "BEARISH":
                bearish += weight

        trend_bull = last_close > float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) and float(ema_slow.iloc[-1]) > float(ema_slow.iloc[-6])
        add("EMA_TREND_ALIGNMENT", trend_bull, "boolean", "BULLISH" if trend_bull else "BEARISH", 25)

        rsi_now = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
        rsi_bull = 50 <= rsi_now <= 70
        add("RSI14", round(rsi_now, 2), "index", "BULLISH" if rsi_bull else ("BEARISH" if rsi_now > 75 or rsi_now < 35 else "NEUTRAL"), 15)

        breakout = last_close > prior_high
        add("20D_BREAKOUT", breakout, "boolean", "BULLISH" if breakout else "NEUTRAL", 20)

        vol_bull = volume_ratio >= cfg.breakout_volume_multiple
        add("VOLUME_RATIO", round(volume_ratio, 2), "x20D_average", "BULLISH" if vol_bull else "NEUTRAL", 15)

        mom_bull = momentum_20 > 0.03
        add("MOMENTUM_20D", round(momentum_20 * 100, 2), "percent", "BULLISH" if mom_bull else ("BEARISH" if momentum_20 < 0 else "NEUTRAL"), 15)

        if pattern:
            evidence.append(EvidenceItem(
                metric_name="PATTERN",
                observed_value=pattern,
                unit="setup",
                source="TechnicalAgent",
                timestamp=decision_time,
                verification_status="VERIFIED",
            ))
            bullish += 10

        total = bullish + bearish
        if bullish >= 55 and bullish > bearish:
            signal = "BULLISH"
        elif bearish >= 45 and bearish > bullish:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        score = float(max(0, min(100, 50 + bullish - bearish)))
        confidence = round(min(1.0, abs(bullish - bearish) / 70), 2) if total else 0.0
        risks: list[str] = []
        if rsi_now > 75:
            risks.append("RSI_OVEREXTENDED")
        if volume_ratio < 1.0:
            risks.append("NO_VOLUME_CONFIRMATION")
        if not trend_bull:
            risks.append("TREND_NOT_ALIGNED")

        metrics = {
            "close": last_close,
            "ema_fast": round(float(ema_fast.iloc[-1]), 4),
            "ema_slow": round(float(ema_slow.iloc[-1]), 4),
            "rsi14": round(rsi_now, 2),
            "atr14": round(float(atr.iloc[-1]), 4) if pd.notna(atr.iloc[-1]) else None,
            "volume_ratio": round(volume_ratio, 3),
            "momentum_20d_pct": round(momentum_20 * 100, 3),
            "prior_20d_high": prior_high,
            "pattern": pattern,
            "pattern_data": pattern_data,
        }
        return cls._result(symbol, run_id, decision_time, signal, score, confidence,
                           metrics, evidence, risks, True, True, None)

    @classmethod
    def _result(cls, symbol: str, run_id: str, decision_time: datetime | date,
                signal: str, score: float, confidence: float | None,
                metrics: dict[str, Any], evidence: list[EvidenceItem], risks: list[str],
                pit_safe: bool, valid: bool, reason: str | None) -> AgentOutput:
        reasons = [reason] if reason else []
        disqualified = not valid
        return AgentOutput(
            agent_name=cls.AGENT_NAME,
            symbol=symbol,
            run_id=run_id,
            timestamp=decision_time if isinstance(decision_time, datetime) else datetime.combine(decision_time, datetime.min.time()),
            signal=signal,
            score=score,
            confidence=confidence,
            metrics=metrics,
            evidence=evidence,
            risks_identified=risks,
            disqualification_triggered=disqualified,
            disqualification_reason=reason,
            execution_time_ms=0,
        )
