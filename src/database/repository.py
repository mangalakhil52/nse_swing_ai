"""
Database Repository Layer.
Provides high-performance data access, bulk upserts, and time-series query operations.
"""

from datetime import date, datetime
import logging
from typing import Any
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.core.models import (
    AgentOutput,
    CandidateScore,
    SymbolMetadata,
    TradeRecommendation,
)
from src.database.schema import (
    AgentOutputModel,
    AgentRunModel,
    CandidateScoreModel,
    OHLCVDailyModel,
    SecurityModel,
    ShadowTradeModel,
    TradeRecommendationModel,
)

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Repository for all database entities."""

    def __init__(self, session: Session):
        self.session = session

    # -----------------------------------------------------------------------
    # Securities Master
    # -----------------------------------------------------------------------

    def upsert_securities(self, securities: list[SymbolMetadata]) -> int:
        """Bulk upserts active securities into the database."""
        count = 0
        for sec in securities:
            stmt = select(SecurityModel).where(SecurityModel.symbol == sec.symbol)
            result = self.session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.company_name = sec.company_name
                existing.isin = sec.isin
                existing.is_fno_eligible = sec.is_fno_eligible
                existing.is_active = sec.is_active
                existing.asm_gsm_stage = sec.asm_gsm_stage
                existing.sector = sec.sector
                existing.industry = sec.industry
            else:
                new_sec = SecurityModel(
                    symbol=sec.symbol,
                    company_name=sec.company_name,
                    isin=sec.isin,
                    exchange=sec.exchange,
                    sector=sec.sector,
                    industry=sec.industry,
                    is_fno_eligible=sec.is_fno_eligible,
                    is_active=sec.is_active,
                    asm_gsm_stage=sec.asm_gsm_stage,
                    lot_size=sec.lot_size,
                )
                self.session.add(new_sec)
            count += 1

        self.session.flush()
        return count

    def get_all_active_securities(self) -> list[SecurityModel]:
        """Retrieves all active trading equities."""
        stmt = select(SecurityModel).where(SecurityModel.is_active.is_(True))
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def get_security_by_symbol(self, symbol: str) -> SecurityModel | None:
        """Fetches security by ticker symbol."""
        stmt = select(SecurityModel).where(SecurityModel.symbol == symbol.upper().strip())
        result = self.session.execute(stmt)
        return result.scalar_one_or_none()

    # -----------------------------------------------------------------------
    # OHLCV Time-Series
    # -----------------------------------------------------------------------

    def save_ohlcv_bars(self, security_id: int, df: pd.DataFrame) -> int:
        """Saves or updates daily OHLCV bars for a security."""
        if df.empty:
            return 0

        saved = 0
        for _, row in df.iterrows():
            bar_time = pd.to_datetime(row["timestamp"]).to_pydatetime()
            stmt = select(OHLCVDailyModel).where(
                OHLCVDailyModel.security_id == security_id,
                OHLCVDailyModel.time == bar_time,
            )
            result = self.session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.open = float(row["open"])
                existing.high = float(row["high"])
                existing.low = float(row["low"])
                existing.close = float(row["close"])
                existing.volume = int(row["volume"])
                existing.delivery_volume = int(row.get("delivery_volume", 0)) if pd.notnull(row.get("delivery_volume")) else None
                existing.delivery_pct = float(row.get("delivery_pct", 0.0)) if pd.notnull(row.get("delivery_pct")) else None
                existing.vwap = float(row.get("vwap", 0.0)) if pd.notnull(row.get("vwap")) else None
                existing.turnover_crores = float(row.get("turnover_crores", 0.0)) if pd.notnull(row.get("turnover_crores")) else None
            else:
                new_bar = OHLCVDailyModel(
                    security_id=security_id,
                    time=bar_time,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    delivery_volume=int(row.get("delivery_volume", 0)) if pd.notnull(row.get("delivery_volume")) else None,
                    delivery_pct=float(row.get("delivery_pct", 0.0)) if pd.notnull(row.get("delivery_pct")) else None,
                    vwap=float(row.get("vwap", 0.0)) if pd.notnull(row.get("vwap")) else None,
                    turnover_crores=float(row.get("turnover_crores", 0.0)) if pd.notnull(row.get("turnover_crores")) else None,
                )
                self.session.add(new_bar)
            saved += 1

        self.session.flush()
        return saved

    def get_ohlcv_dataframe(self, symbol: str, lookback_days: int = 150) -> pd.DataFrame:
        """Retrieves stored OHLCV bars as a pandas DataFrame."""
        sec = self.get_security_by_symbol(symbol)
        if not sec:
            return pd.DataFrame()

        stmt = (
            select(OHLCVDailyModel)
            .where(OHLCVDailyModel.security_id == sec.id)
            .order_by(desc(OHLCVDailyModel.time))
            .limit(lookback_days)
        )
        result = self.session.execute(stmt)
        bars = list(result.scalars().all())
        if not bars:
            return pd.DataFrame()

        records = [
            {
                "timestamp": b.time,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "delivery_volume": b.delivery_volume,
                "delivery_pct": b.delivery_pct,
                "vwap": b.vwap,
                "turnover_crores": b.turnover_crores,
            }
            for b in reversed(bars)
        ]
        return pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Agent Runs & Outputs
    # -----------------------------------------------------------------------

    def create_agent_run(
        self,
        run_id: str,
        market_regime: str,
        universe_size: int,
        quant_candidates_count: int,
    ) -> AgentRunModel:
        """Initializes a new master agent execution cycle log."""
        run = AgentRunModel(
            id=run_id,
            started_at=datetime.utcnow(),
            status="RUNNING",
            market_regime=market_regime,
            universe_size=universe_size,
            quant_candidates_count=quant_candidates_count,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def complete_agent_run(
        self,
        run_id: str,
        status: str,
        recommended_count: int,
        log_text: str | None = None,
    ) -> None:
        """Marks agent run cycle as completed."""
        stmt = select(AgentRunModel).where(AgentRunModel.id == run_id)
        result = self.session.execute(stmt)
        run = result.scalar_one_or_none()
        if run:
            run.completed_at = datetime.utcnow()
            run.status = status
            run.recommended_count = recommended_count
            run.log_text = log_text
            self.session.flush()

    def save_agent_output(self, output: AgentOutput) -> None:
        """Stores structured output from a specialist agent."""
        record = AgentOutputModel(
            run_id=output.run_id,
            symbol=output.symbol,
            agent_name=output.agent_name,
            status=output.status.value,
            signal=output.signal.value,
            score=output.score,
            confidence=output.confidence,
            disqualification_triggered=output.disqualification_triggered,
            disqualification_reason=output.disqualification_reason,
            metrics_json=output.metrics,
            evidence_json=[e.model_dump(mode="json") for e in output.evidence],
            risks_json=output.risks_identified,
            execution_time_ms=output.execution_time_ms,
        )
        self.session.add(record)
        self.session.flush()

    def save_candidate_score(self, score: CandidateScore) -> None:
        """Stores calculated 100-pt composite candidate score."""
        record = CandidateScoreModel(
            run_id=score.run_id,
            symbol=score.symbol,
            composite_score=score.composite_score,
            conviction_grade=score.conviction_grade.value,
            confluence_state=score.confluence_state.value,
            factor_scores_json=score.factor_scores,
            passed_risk_veto=score.passed_risk_veto,
            rejection_reasons=score.rejection_reasons,
        )
        self.session.add(record)
        self.session.flush()

    def save_trade_recommendation(self, rec: TradeRecommendation) -> None:
        """Persists official actionable trade recommendation."""
        record = TradeRecommendationModel(
            recommendation_id=rec.recommendation_id,
            run_id=rec.run_id,
            symbol=rec.symbol,
            recommendation_date=rec.recommendation_date,
            conviction=rec.conviction.value,
            current_market_price=rec.levels.current_market_price,
            entry_trigger_price=rec.levels.entry_trigger_price,
            stop_loss_price=rec.levels.stop_loss_price,
            risk_percentage=rec.levels.risk_percentage,
            target_1=rec.levels.target_1,
            target_2=rec.levels.target_2,
            target_3=rec.levels.target_3,
            risk_reward_t1=rec.levels.risk_reward_t1,
            risk_reward_t2=rec.levels.risk_reward_t2,
            position_size_shares=rec.levels.position_size_shares,
            trade_dossier_json=rec.model_dump(mode="json"),
            status=rec.status.value,
        )
        self.session.add(record)

        # Create shadow trade ledger entry
        shadow_trade = ShadowTradeModel(
            recommendation_id=rec.recommendation_id,
            symbol=rec.symbol,
            entry_date=rec.recommendation_date,
            entry_price=rec.levels.entry_trigger_price,
            status="ACTIVE",
        )
        self.session.add(shadow_trade)
        self.session.flush()
