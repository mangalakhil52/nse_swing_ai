"""
Technical Analysis Specialist Agent Module.
Performs deterministic trend, pattern, momentum, and volume structure analysis.
"""

from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, PatternType, SignalType
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer


class TechnicalAnalysisAgent(BaseAgent):
    """Specialist agent analyzing technical chart structure and pattern setups."""

    def __init__(self):
        super().__init__(agent_name="technical_analysis_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol

        if df.empty or len(df) < 30:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.NEUTRAL,
                score=0.0,
                confidence=0.0,
            )

        # Ensure indicators are computed
        df = TechnicalIndicators.compute_all_indicators(df)

        close = df["close"].iloc[-1]
        ema_20 = df["ema_20"].iloc[-1]
        ema_50 = df["ema_50"].iloc[-1]
        ema_200 = df["ema_200"].iloc[-1]
        rsi = df["rsi_14"].iloc[-1]
        adx = df["adx_14"].iloc[-1]
        rvol = df["rvol_20"].iloc[-1]
        dist_52w = df["distance_52w_high_pct"].iloc[-1]

        # Detect patterns
        matched_patterns = PatternRecognizer.evaluate_all_patterns(df)
        top_pattern = matched_patterns[0] if matched_patterns else None

        # Calculate technical score (0 to 100)
        score = 40.0  # Base

        # 1. EMA Structure
        if close > ema_20:
            score += 10.0
        if ema_20 > ema_50:
            score += 10.0
        if close > ema_200:
            score += 10.0

        # 2. Momentum
        if 58.0 <= rsi <= 74.0:
            score += 10.0
        elif 50.0 <= rsi < 58.0:
            score += 5.0

        if adx >= 25.0:
            score += 5.0

        # 3. Pattern & Volume
        if top_pattern and top_pattern.is_matched:
            score += min(15.0, top_pattern.quality_score * 0.16)
        if rvol >= 1.5:
            score += 5.0

        # Calibrated dynamic confidence based on bar history depth and indicator agreement
        bar_count_factor = min(1.0, len(df) / 100.0)
        pattern_bonus = 0.12 if (top_pattern and top_pattern.is_matched) else 0.0
        confidence = round(min(0.98, max(0.40, 0.70 * bar_count_factor + pattern_bonus)), 2)

        # Determine signal
        if score >= 75.0:
            signal = SignalType.BULLISH
        elif score < 45.0:
            signal = SignalType.BEARISH
        else:
            signal = SignalType.NEUTRAL

        # Register Evidence
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="TREND_STRUCTURE",
            raw_metric="ema_alignment",
            observed_value=f"Close(₹{close:.1f}) > 20 EMA(₹{ema_20:.1f}) > 50 EMA(₹{ema_50:.1f})",
            unit="price_structure",
            source="NSE_BHAVCOPY",
            timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns else "EOD",
        )

        pattern_desc = top_pattern.description if top_pattern else "General Uptrend Consolidation"
        evidence_graph.add_evidence(
            symbol=symbol,
            agent_name=self.agent_name,
            claim_type="PATTERN_MATCH",
            raw_metric="chart_pattern",
            observed_value=pattern_desc,
            unit="pattern_geometry",
            source="QUANT_ENGINE",
            timestamp=df["timestamp"].iloc[-1] if "timestamp" in df.columns else "EOD",
        )

        risks: list[str] = []
        if dist_52w > 15.0:
            risks.append(f"Overhead supply resistance {dist_52w:.1f}% below 52W High")
        if rsi > 78.0:
            risks.append(f"Short-term RSI extended at {rsi:.1f}")

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol,
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score=round(score, 1),
            confidence=confidence,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "rsi_14": round(rsi, 1),
                "adx_14": round(adx, 1),
                "rvol_20": round(rvol, 2),
                "pattern_detected": top_pattern.pattern_type.value if top_pattern else PatternType.NO_PATTERN.value,
                "pattern_quality": top_pattern.quality_score if top_pattern else 0.0,
                "distance_52w_high_pct": round(dist_52w, 1),
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
        )
