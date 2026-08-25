"""Technical Analysis Specialist Agent — P0 #14C."""
from datetime import date, datetime
from typing import Any
import pandas as pd
from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, PatternType, SignalType
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer
from src.data.data_quality import DataQualityGate, DataQualityStatus
from src.data.point_in_time import PointInTimeFilter
from src.architecture.contracts import AgentAnalysisResult, StructuredEvidence


class TechnicalAnalysisAgent(BaseAgent):
    """Deterministic technical specialist; no downstream trade construction."""
    def __init__(self):
        super().__init__(agent_name="technical_analysis_agent")

    @staticmethod
    def _decision_time(context: dict[str, Any], df: pd.DataFrame) -> datetime | date:
        value = context.get("decision_time") or context.get("as_of_date")
        if value is not None:
            return value
        if "timestamp" in df.columns and not df.empty:
            return pd.to_datetime(df["timestamp"].iloc[-1]).to_pydatetime()
        raise ValueError("decision_time is required for TechnicalAnalysisAgent")

    @staticmethod
    def _pit_slice(df: pd.DataFrame, decision_time: datetime | date) -> pd.DataFrame:
        return PointInTimeFilter.filter_market_data(df, decision_time)

    async def _analyze(self, symbol_meta: SymbolMetadata, df: pd.DataFrame,
                       evidence_graph: EvidenceGraph, run_id: str,
                       context: dict[str, Any]) -> AgentOutput:
        symbol = symbol_meta.symbol.upper().strip()
        decision_time = self._decision_time(context, df)
        df = self._pit_slice(df.copy(), decision_time)
        if df.empty:
            return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE, signal=SignalType.UNKNOWN, score=0.0,
                confidence=0.0, data_freshness=DataFreshness.UNKNOWN,
                risks_identified=["NO_OHLCV_DATA_AT_DECISION_TIME"])

        dq = DataQualityGate.evaluate_ohlcv(df, symbol, decision_time, min_required_bars=30)
        if dq.status in (DataQualityStatus.INVALID, DataQualityStatus.PIT_VIOLATION):
            return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE, signal=SignalType.UNKNOWN, score=0.0,
                confidence=0.0, data_freshness=DataFreshness.UNKNOWN,
                metrics={"pit_safe": dq.pit_safe, "data_quality_status": dq.status.value},
                risks_identified=list(dq.reasons), disqualification_triggered=True,
                disqualification_reason="TECHNICAL_DATA_QUALITY_FAILURE")
        if len(df) < 30:
            return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE, signal=SignalType.UNKNOWN, score=0.0,
                confidence=0.0, data_freshness=DataFreshness.UNKNOWN,
                metrics={"bars": len(df), "pit_safe": dq.pit_safe},
                risks_identified=["INSUFFICIENT_HISTORY"], disqualification_triggered=True,
                disqualification_reason="INSUFFICIENT_HISTORY")

        df = TechnicalIndicators.compute_all_indicators(df)
        close = float(df["close"].iloc[-1]); ema_20 = float(df["ema_20"].iloc[-1])
        ema_50 = float(df["ema_50"].iloc[-1]); ema_200 = float(df["ema_200"].iloc[-1])
        rsi = float(df["rsi_14"].iloc[-1]); adx = float(df["adx_14"].iloc[-1])
        rvol = float(df["rvol_20"].iloc[-1]); dist_52w = float(df["distance_52w_high_pct"].iloc[-1])
        matched_patterns = PatternRecognizer.evaluate_all_patterns(df)
        top_pattern = matched_patterns[0] if matched_patterns else None

        # Preserve the existing deterministic specialist score; this is NOT final conviction.
        raw_score = 40.0
        if close > ema_20: raw_score += 10.0
        if ema_20 > ema_50: raw_score += 10.0
        if close > ema_200: raw_score += 10.0
        if 58.0 <= rsi <= 74.0: raw_score += 10.0
        elif 50.0 <= rsi < 58.0: raw_score += 5.0
        if adx >= 25.0: raw_score += 5.0
        if top_pattern and top_pattern.is_matched: raw_score += min(15.0, top_pattern.quality_score * 0.16)
        if rvol >= 1.5: raw_score += 5.0
        # AgentOutput.score is contractually bounded to [0, 100]. The component
        # bonuses above can legitimately total slightly above 100, so clamp rather
        # than allowing an otherwise valid technical analysis to fail validation.
        score = min(100.0, max(0.0, raw_score))

        bar_factor = min(1.0, len(df) / 100.0)
        pattern_bonus = 0.12 if (top_pattern and top_pattern.is_matched) else 0.0
        confidence = round(min(0.98, max(0.40, 0.70 * bar_factor + pattern_bonus)), 2)
        signal = SignalType.BULLISH if score >= 75.0 else (SignalType.BEARISH if score < 45.0 else SignalType.NEUTRAL)
        timestamp = df["timestamp"].iloc[-1] if "timestamp" in df.columns else decision_time

        evidence_graph.add_evidence(symbol=symbol, agent_name=self.agent_name,
            claim_type="TREND_STRUCTURE", raw_metric="ema_alignment",
            observed_value=f"Close={close:.2f}; EMA20={ema_20:.2f}; EMA50={ema_50:.2f}; EMA200={ema_200:.2f}",
            unit="price_structure", source="NSE_BHAVCOPY", timestamp=timestamp)
        pattern_desc = top_pattern.description if top_pattern else "General Uptrend Consolidation"
        evidence_graph.add_evidence(symbol=symbol, agent_name=self.agent_name,
            claim_type="PATTERN_MATCH", raw_metric="chart_pattern", observed_value=pattern_desc,
            unit="pattern_geometry", source="QUANT_ENGINE", timestamp=timestamp)

        risks = []
        if dist_52w > 15.0: risks.append(f"Overhead supply resistance {dist_52w:.1f}% below 52W High")
        if rsi > 78.0: risks.append(f"Short-term RSI extended at {rsi:.1f}")

        return AgentOutput(agent_name=self.agent_name, symbol=symbol, run_id=run_id,
            status=AgentStatus.SUCCESS, signal=signal, score=round(score, 1), confidence=confidence,
            data_freshness=DataFreshness.RECENT,
            metrics={"rsi_14": round(rsi,1), "adx_14": round(adx,1), "rvol_20": round(rvol,2),
                "pattern_detected": top_pattern.pattern_type.value if top_pattern else PatternType.NO_PATTERN.value,
                "pattern_quality": top_pattern.quality_score if top_pattern else 0.0,
                "distance_52w_high_pct": round(dist_52w,1), "pit_safe": bool(dq.pit_safe),
                "decision_time": decision_time.isoformat()},
            evidence=evidence_graph.to_evidence_items(symbol), risks_identified=risks)

    async def analyze_contract(self, symbol_meta: SymbolMetadata, df: pd.DataFrame,
                               decision_time: datetime | date, run_id: str = "") -> AgentAnalysisResult:
        """Emit the #14A AgentAnalysisResult contract with explicit PIT status."""
        symbol = symbol_meta.symbol.upper().strip()
        pit_df = self._pit_slice(df, decision_time)
        dq = DataQualityGate.evaluate_ohlcv(pit_df, symbol, decision_time, min_required_bars=30)
        if pit_df.empty or dq.status in (DataQualityStatus.INVALID, DataQualityStatus.PIT_VIOLATION):
            return AgentAnalysisResult(symbol=symbol, agent_name=self.agent_name, decision_time=decision_time,
                signal=SignalType.UNKNOWN, score=0.0, confidence=0.0, data_quality=dq, pit_safe=False,
                status=AgentStatus.DATA_UNAVAILABLE,
                reasons=list(dq.reasons) or ["NO_OHLCV_DATA_AT_DECISION_TIME"])

        graph = EvidenceGraph()
        output = await self._analyze(symbol_meta, pit_df, graph, run_id, {"decision_time": decision_time})
        structured = [StructuredEvidence(source="TECHNICAL",
            observation=f"{item.metric_name}: {item.observed_value}", as_of=decision_time,
            direction=output.signal,
            strength="HIGH" if output.score >= 75 else ("MEDIUM" if output.score >= 55 else "LOW"),
            reliability=output.confidence or 0.0, pit_safe=True)
            for item in output.evidence]
        return AgentAnalysisResult(symbol=symbol, agent_name=self.agent_name, decision_time=decision_time,
            signal=output.signal, score=output.score, confidence=output.confidence or 0.0,
            evidence=structured, risks=output.risks_identified, data_quality=dq, pit_safe=True,
            status=output.status, reasons=output.risks_identified)
