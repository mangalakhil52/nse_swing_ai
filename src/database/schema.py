"""
SQLAlchemy 2.0 ORM Schema Models for nse_swing_ai.
Defines all persistent relational entities for market data, indicators, agent logs, scores, and trades.
Uses explicit datetime module typing to prevent namespace collision in Python 3.13+.
"""

import datetime as dt
from typing import Any
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy entities."""
    pass


class SecurityModel(Base):
    __tablename__ = "securities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), unique=True, nullable=False, index=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    isin: Mapped[str | None] = mapped_column(String(12), unique=True, nullable=True)
    exchange: Mapped[str] = mapped_column(String(10), default="NSE")
    sector: Mapped[str | None] = mapped_column(String(100), default="General")
    industry: Mapped[str | None] = mapped_column(String(100), default="General")
    is_fno_eligible: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    asm_gsm_stage: Mapped[int] = mapped_column(Integer, default=0)
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    # Relationships
    daily_bars: Mapped[list["OHLCVDailyModel"]] = relationship("OHLCVDailyModel", back_populates="security", cascade="all, delete-orphan")
    fundamentals: Mapped[list["FundamentalModel"]] = relationship("FundamentalModel", back_populates="security", cascade="all, delete-orphan")


class OHLCVDailyModel(Base):
    __tablename__ = "ohlcv_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(Integer, ForeignKey("securities.id", ondelete="CASCADE"), nullable=False, index=True)
    time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    delivery_volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    delivery_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover_crores: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_source: Mapped[str] = mapped_column(String(50), default="NSE_BHAVCOPY")

    __table_args__ = (
        UniqueConstraint("security_id", "time", name="uq_security_time"),
    )

    security: Mapped["SecurityModel"] = relationship("SecurityModel", back_populates="daily_bars")


class TechnicalIndicatorModel(Base):
    __tablename__ = "technical_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(Integer, ForeignKey("securities.id", ondelete="CASCADE"), nullable=False, index=True)
    time: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)
    ema_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_200: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    mansfield_rs: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance_52w_high_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_volume_20d: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("security_id", "time", name="uq_indicator_security_time"),
    )


class MarketRegimeModel(Base):
    __tablename__ = "market_regimes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    regime_date: Mapped[dt.date] = mapped_column("date", Date, unique=True, nullable=False, index=True)
    index_symbol: Mapped[str] = mapped_column(String(30), default="NIFTY 50")
    close: Mapped[float] = mapped_column(Float, nullable=False)
    regime: Mapped[str] = mapped_column(String(30), nullable=False)
    trading_stance: Mapped[str] = mapped_column(String(30), nullable=False)
    advance_decline_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_above_50_sma: Mapped[float | None] = mapped_column(Float, nullable=True)
    india_vix: Mapped[float | None] = mapped_column(Float, nullable=True)
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class FundamentalModel(Base):
    __tablename__ = "fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(Integer, ForeignKey("securities.id", ondelete="CASCADE"), nullable=False, index=True)
    period_end_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    sales_growth_yoy: Mapped[float | None] = mapped_column(Float, nullable=True)
    pat_growth_yoy: Mapped[float | None] = mapped_column(Float, nullable=True)
    ebitda_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    roce: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    cfo_to_pat: Mapped[float | None] = mapped_column(Float, nullable=True)
    promoter_holding_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    promoter_pledging_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fii_holding_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dii_holding_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fundamental_grade: Mapped[str | None] = mapped_column(String(20), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("security_id", "period_end_date", name="uq_security_period"),
    )

    security: Mapped["SecurityModel"] = relationship("SecurityModel", back_populates="fundamentals")


class NewsArticleModel(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    publisher: Mapped[str] = mapped_column(String(100), nullable=False)
    source_tier: Mapped[int] = mapped_column(Integer, default=2)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False, index=True)
    sentiment: Mapped[str] = mapped_column(String(20), default="NEUTRAL")
    materiality_score: Mapped[float] = mapped_column(Float, default=0.5)
    is_catalyst: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class AgentRunModel(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="RUNNING")
    market_regime: Mapped[str | None] = mapped_column(String(30), nullable=True)
    universe_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quant_candidates_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    researched_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recommended_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_version: Mapped[str] = mapped_column(String(30), default="v1.0.0")
    log_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentOutputModel(Base):
    __tablename__ = "agent_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(50), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(60), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="SUCCESS")
    signal: Mapped[str | None] = mapped_column(String(30), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    disqualification_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    disqualification_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    evidence_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    risks_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    execution_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class CandidateScoreModel(Base):
    __tablename__ = "candidate_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(50), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    conviction_grade: Mapped[str] = mapped_column(String(10), nullable=False)
    confluence_state: Mapped[str] = mapped_column(String(40), nullable=False)
    factor_scores_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    passed_risk_veto: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class TradeRecommendationModel(Base):
    __tablename__ = "trade_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_id: Mapped[str] = mapped_column(String(60), unique=True, nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(50), ForeignKey("agent_runs.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    recommendation_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    conviction: Mapped[str] = mapped_column(String(10), nullable=False)
    current_market_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_trigger_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_price: Mapped[float] = mapped_column(Float, nullable=False)
    risk_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    target_1: Mapped[float] = mapped_column(Float, nullable=False)
    target_2: Mapped[float] = mapped_column(Float, nullable=False)
    target_3: Mapped[float] = mapped_column(Float, nullable=False)
    risk_reward_t1: Mapped[float] = mapped_column(Float, nullable=False)
    risk_reward_t2: Mapped[float] = mapped_column(Float, nullable=False)
    position_size_shares: Mapped[int] = mapped_column(Integer, nullable=False)
    trade_dossier_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="PENDING_ENTRY")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ShadowTradeModel(Base):
    __tablename__ = "shadow_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_id: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    entry_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pnl_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_rupees: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_adverse_excursion_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_favorable_excursion_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_sessions: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="ACTIVE")
