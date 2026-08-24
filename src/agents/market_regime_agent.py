"""Market-regime specialist agent — P0/P1 PIT-safe adapter around MarketRegimeClassifier."""
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.architecture.contracts import AgentAnalysisResult, StructuredEvidence
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, MarketRegime, SignalType
from src.data.data_quality import DataQualityGate, DataQualityStatus
from src.data.point_in_time import PointInTimeFilter
from src.quant.regime import MarketRegimeClassifier


class MarketRegimeAgent(BaseAgent):
    """Specialist desk for market-wide regime and risk posture.

    This agent classifies the NIFTY regime but never constructs a trade.  It
    requires an explicit decision timestamp for contract-level execution and
    fails closed when required market observations are unavailable.
    """

    def __init__(self) -> None:
        super().__init__(agent_name="market_regime_agent")

    @staticmethod
    def _decision_time(context: dict[str, Any]) -> date | datetime:
        value = context.get("decision_time") or context.get("as_of_datetime") or context.get("as_of_date")
        if value is None:
            raise ValueError("decision_time is required for MarketRegimeAgent")
        return value if isinstance(value, (date, datetime)) else pd.to_datetime(value).to_pydatetime()

    @staticmethod
    def _pit_nifty(df: pd.DataFrame | None, decision_time: date | datetime) -> pd.DataFrame | None:
        if df is None:
            return None
        pit = PointInTimeFilter.filter_market_data(df, decision_time)
        PointInTimeFilter.enforce_pit_boundary(pit, decision_time)
        return pit

    @staticmethod
    def _signal(regime: MarketRegime) -> SignalType:
        if regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
            return SignalType.BULLISH
        if regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR):
            return SignalType.BEARISH
        if regime == MarketRegime.NEUTRAL:
            return SignalType.NEUTRAL
        return SignalType.UNKNOWN

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        decision_time = self._decision_time(context)
        nifty_df = self._pit_nifty(context.get("nifty_df"), decision_time)
        if nifty_df is None or nifty_df.empty:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol_meta.symbol.upper().strip(),
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.UNKNOWN,
                score=0.0,
                confidence=0.0,
                data_freshness=DataFreshness.UNKNOWN,
                metrics={"decision_time": decision_time.isoformat()},
                risks_identified=["NIFTY_BENCHMARK_UNAVAILABLE"],
                disqualification_triggered=True,
                disqualification_reason="MARKET_REGIME_DATA_UNAVAILABLE",
            )

        result = MarketRegimeClassifier.classify_regime(
            nifty_df=nifty_df,
            advance_decline_ratio=context.get("advance_decline_ratio"),
            pct_above_50_sma=context.get("pct_above_50_sma"),
            india_vix=context.get("india_vix"),
            as_of_date=decision_time,
        )
        if result.regime == MarketRegime.UNKNOWN:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol_meta.symbol.upper().strip(),
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.UNKNOWN,
                score=0.0,
                confidence=0.0,
                data_freshness=DataFreshness.UNKNOWN,
                metrics={
                    "regime": result.regime.value,
                    "trading_stance": result.trading_stance.value,
                    "risk_multiplier": result.risk_multiplier,
                },
                risks_identified=["REQUIRED_MARKET_REGIME_INPUT_MISSING"],
                disqualification_triggered=True,
                disqualification_reason="MARKET_REGIME_DATA_UNAVAILABLE",
            )

        signal = self._signal(result.regime)
        evidence_graph.add_evidence(
            symbol=symbol_meta.symbol.upper().strip(),
            agent_name=self.agent_name,
            claim_type="MARKET_REGIME",
            raw_metric="regime",
            observed_value=result.regime.value,
            unit="classification",
            source="MARKET_REGIME_CLASSIFIER",
            timestamp=decision_time,
        )
        evidence_graph.add_evidence(
            symbol=symbol_meta.symbol.upper().strip(),
            agent_name=self.agent_name,
            claim_type="MARKET_REGIME",
            raw_metric="risk_multiplier",
            observed_value=result.risk_multiplier,
            unit="position_risk_multiplier",
            source="MARKET_REGIME_CLASSIFIER",
            timestamp=decision_time,
        )

        return AgentOutput(
            agent_name=self.agent_name,
            symbol=symbol_meta.symbol.upper().strip(),
            run_id=run_id,
            status=AgentStatus.SUCCESS,
            signal=signal,
            score={
                MarketRegime.STRONG_BULL: 100.0,
                MarketRegime.BULL: 75.0,
                MarketRegime.NEUTRAL: 50.0,
                MarketRegime.BEAR: 25.0,
                MarketRegime.STRONG_BEAR: 0.0,
            }[result.regime],
            confidence=result.confidence,
            data_freshness=DataFreshness.RECENT,
            metrics={
                "regime": result.regime.value,
                "trading_stance": result.trading_stance.value,
                "nifty_close": result.nifty_close,
                "advance_decline_ratio": result.advance_decline_ratio,
                "pct_above_50_sma": result.pct_above_50_sma,
                "india_vix": result.india_vix,
                "allow_long_swing_trades": result.allow_long_swing_trades,
                "risk_multiplier": result.risk_multiplier,
                "decision_time": decision_time.isoformat(),
            },
            evidence=evidence_graph.to_evidence_items(symbol_meta.symbol.upper().strip()),
            risks_identified=[] if result.allow_long_swing_trades else [result.trend_description],
        )

    async def analyze_contract(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        decision_time: datetime | date,
        run_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> AgentAnalysisResult:
        """Emit the common #14A contract with explicit PIT safety."""
        context = dict(context or {})
        context["decision_time"] = decision_time
        nifty_df = self._pit_nifty(context.get("nifty_df"), decision_time)
        if nifty_df is None or nifty_df.empty:
            return AgentAnalysisResult(
                symbol=symbol_meta.symbol.upper().strip(),
                agent_name=self.agent_name,
                decision_time=decision_time,
                signal=SignalType.UNKNOWN,
                score=0.0,
                confidence=0.0,
                pit_safe=False,
                status=AgentStatus.DATA_UNAVAILABLE,
                reasons=["NIFTY_BENCHMARK_UNAVAILABLE"],
            )

        # Validate benchmark quality independently before classification.
        dq = DataQualityGate.evaluate_ohlcv(
            nifty_df,
            "NIFTY50",
            decision_time,
            min_required_bars=50,
        )
        if dq.status in (DataQualityStatus.INVALID, DataQualityStatus.PIT_VIOLATION):
            return AgentAnalysisResult(
                symbol=symbol_meta.symbol.upper().strip(),
                agent_name=self.agent_name,
                decision_time=decision_time,
                signal=SignalType.UNKNOWN,
                score=0.0,
                confidence=0.0,
                data_quality=dq,
                pit_safe=False,
                status=AgentStatus.DATA_UNAVAILABLE,
                reasons=list(dq.reasons),
            )

        graph = EvidenceGraph(run_id)
        output = await self._analyze(symbol_meta, df, graph, run_id, context)
        pit_safe = output.status == AgentStatus.SUCCESS and bool(dq.pit_safe)
        signal = output.signal if pit_safe else SignalType.UNKNOWN
        structured = [
            StructuredEvidence(
                source="MARKET_REGIME",
                observation=f"{item.metric_name}: {item.observed_value}",
                as_of=decision_time,
                direction=signal,
                strength="HIGH" if output.score >= 75 else ("MEDIUM" if output.score >= 50 else "LOW"),
                reliability=output.confidence if pit_safe else 0.0,
                pit_safe=pit_safe,
            )
            for item in output.evidence
        ]
        return AgentAnalysisResult(
            symbol=symbol_meta.symbol.upper().strip(),
            agent_name=self.agent_name,
            decision_time=decision_time,
            signal=signal,
            score=output.score if pit_safe else 0.0,
            confidence=output.confidence if pit_safe else 0.0,
            evidence=structured,
            risks=output.risks_identified,
            data_quality=dq,
            pit_safe=pit_safe,
            status=output.status if pit_safe else AgentStatus.DATA_UNAVAILABLE,
            reasons=output.risks_identified if pit_safe else ["MARKET_REGIME_UNAVAILABLE_OR_PIT_UNVERIFIED"],
        )
