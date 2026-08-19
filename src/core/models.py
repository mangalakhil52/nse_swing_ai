"""
Core Pydantic Data Models & Agent JSON Communication Contracts.
Provides strict validation, deterministic serialization, and full type safety across all system layers.
"""

from datetime import date, datetime
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from src.core.types import (
    ConvictionGrade,
    ConfluenceState,
    AgentStatus,
    SignalType,
    SourceTier,
    DataFreshness,
    SentimentType,
    CatalystType,
    FundamentalGrade,
    ForensicVerdict,
    TradeStatus,
    ExitReason,
)


class BaseDataModel(BaseModel):
    """Base model enforcing strict configuration."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True
    )


# ---------------------------------------------------------------------------
# Universe & Market Master Models
# ---------------------------------------------------------------------------

class SymbolMetadata(BaseDataModel):
    symbol: str = Field(..., description="NSE Trading Symbol (e.g. 'TRENT', 'RELIANCE')")
    company_name: str = Field(..., description="Full legal corporate name")
    isin: str | None = Field(default=None, description="ISIN Code (e.g. 'INE849A01020')")
    exchange: str = Field(default="NSE")
    sector: str = Field(default="Unknown", description="Broad Sector Category (e.g. 'Retail', 'IT')")
    industry: str = Field(default="Unknown", description="Specific Industry (e.g. 'Department Stores')")
    is_fno_eligible: bool = Field(default=False, description="Whether symbol trades in NSE F&O segment")
    is_active: bool = Field(default=True, description="Whether active for equity trading")
    asm_gsm_stage: int = Field(default=0, description="SEBI Surveillance Stage: 0=None, 1-4=ASM/GSM")
    lot_size: int = Field(default=1, description="F&O lot size if applicable, else 1")


# ---------------------------------------------------------------------------
# Market Data Models (OHLCV, Quotes, Breadth)
# ---------------------------------------------------------------------------

class OHLCVCandle(BaseDataModel):
    timestamp: datetime = Field(..., description="Candle market date/time (IST)")
    symbol: str
    open: float = Field(..., gt=0.0)
    high: float = Field(..., gt=0.0)
    low: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    volume: int = Field(..., ge=0)
    delivery_volume: int | None = Field(default=None, ge=0)
    delivery_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    vwap: float | None = Field(default=None)
    turnover_crores: float | None = Field(default=None)
    data_source: str = Field(default="NSE_BHAVCOPY")


class DailyBarSeries(BaseDataModel):
    symbol: str
    bars: list[OHLCVCandle]
    source: str = Field(default="NSE_BHAVCOPY")
    freshness: DataFreshness = Field(default=DataFreshness.RECENT)


class LiveQuote(BaseDataModel):
    symbol: str
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    prev_close: float
    change_pct: float
    total_traded_volume: int
    total_traded_value_crores: float
    upper_circuit_limit: float
    lower_circuit_limit: float
    vwap: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    spread_pct: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data_source: str = Field(default="NSE_LIVE_API")


class MarketBreadthData(BaseDataModel):
    date: date
    index_symbol: str = Field(default="NIFTY 500")
    advances: int = Field(..., ge=0)
    declines: int = Field(..., ge=0)
    unchanged: int = Field(default=0, ge=0)
    advance_decline_ratio: float = Field(..., ge=0.0)
    pct_stocks_above_20_ema: float | None = None
    pct_stocks_above_50_sma: float | None = None
    pct_stocks_above_200_sma: float | None = None
    india_vix: float | None = None


# ---------------------------------------------------------------------------
# Fundamental & Intelligence Models
# ---------------------------------------------------------------------------

class QuarterlyFinancials(BaseDataModel):
    symbol: str
    period_end_date: date
    filing_date: date | None = Field(default=None, description="Actual regulatory filing date")
    available_at: date | None = Field(default=None, description="Public availability timestamp")
    sales_crores: float
    sales_growth_yoy_pct: float
    pat_crores: float
    pat_growth_yoy_pct: float
    ebitda_margin_pct: float
    eps_inr: float
    data_source: str = Field(default="SCREENER_API")
    pit_status: str = Field(default="PIT_UNVERIFIED")


class AnnualRatios(BaseDataModel):
    symbol: str
    fiscal_year: int
    roe_pct: float
    roce_pct: float
    debt_to_equity: float
    cfo_crores: float
    cfo_to_pat_ratio: float
    working_capital_days: float | None = None
    fundamental_grade: FundamentalGrade = Field(default=FundamentalGrade.GOOD)
    available_at: date | None = Field(default=None)


class ShareholdingPattern(BaseDataModel):
    symbol: str
    quarter_date: date
    promoter_pct: float = Field(..., ge=0.0, le=100.0)
    promoter_pledged_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    fii_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    dii_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    public_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    promoter_change_quarterly_pct: float = Field(default=0.0)
    available_at: date | None = Field(default=None)


class NewsArticle(BaseDataModel):
    symbol: str
    headline: str
    summary: str | None = None
    publisher: str
    source_tier: SourceTier = Field(default=SourceTier.TIER_2)
    source_url: str | None = None
    published_at: datetime
    sentiment: SentimentType = Field(default=SentimentType.NEUTRAL)
    materiality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_catalyst: bool = Field(default=False)
    catalyst_type: CatalystType = Field(default=CatalystType.NO_CATALYST)
    extraction_reasoning: str | None = None


class CorporateAnnouncement(BaseDataModel):
    symbol: str
    headline: str
    category: str
    broadcast_timestamp: datetime
    available_at: datetime | None = Field(default=None)
    exchange: str = "NSE"
    attachment_url: str | None = None


class CorporateEvent(BaseDataModel):
    symbol: str
    event_type: str = Field(..., description="e.g. 'BOARD_MEETING_RESULTS', 'DIVIDEND'")
    event_date: date
    announcement_date: date | None = Field(default=None)
    available_at: date | None = Field(default=None)
    purpose: str


class SplitBonusAdjustment(BaseDataModel):
    symbol: str
    adjustment_type: str = Field(..., description="'SPLIT' or 'BONUS'")
    record_date: date
    ex_date: date
    multiplier: float


# ---------------------------------------------------------------------------
# Evidence & Agent Communication Contracts
# ---------------------------------------------------------------------------

class EvidenceItem(BaseDataModel):
    metric_name: str
    observed_value: Any
    unit: str
    source: str
    timestamp: datetime | str
    verification_status: str = Field(default="VERIFIED")
    citation_url: str | None = None


class AgentOutput(BaseDataModel):
    agent_name: str
    symbol: str
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: AgentStatus = Field(default=AgentStatus.SUCCESS)
    signal: SignalType = Field(default=SignalType.NEUTRAL)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    data_freshness: DataFreshness = Field(default=DataFreshness.RECENT)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    risks_identified: list[str] = Field(default_factory=list)
    disqualification_triggered: bool = Field(default=False)
    disqualification_reason: str | None = None
    execution_time_ms: int = Field(default=0)


class CandidateScore(BaseDataModel):
    symbol: str
    run_id: str
    composite_score: float = Field(..., ge=0.0, le=100.0)
    conviction_grade: ConvictionGrade
    confluence_state: ConfluenceState
    factor_scores: dict[str, float]
    passed_risk_veto: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class TradeLevels(BaseDataModel):
    symbol: str
    current_market_price: float
    entry_trigger_price: float
    stop_loss_price: float
    risk_rupees: float
    risk_percentage: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward_t1: float
    risk_reward_t2: float
    risk_reward_t3: float
    position_size_shares: int
    allocated_capital_rupees: float
    expected_holding_period: str = "3-15 sessions"
    invalidation_criteria: str


class TradeRecommendation(BaseDataModel):
    recommendation_id: str
    run_id: str
    symbol: str
    company_name: str
    sector: str
    recommendation_date: date
    conviction: ConvictionGrade
    composite_score: float
    levels: TradeLevels
    technical_setup_description: str
    catalyst_summary: str
    fundamental_summary: str
    sector_context: str
    market_regime: str
    major_risks: list[str]
    invalidation_rules: str
    why_this_trade: list[str]
    evidence_dossier: list[EvidenceItem]
    status: TradeStatus = Field(default=TradeStatus.PENDING_ENTRY)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ShadowTradeRecord(BaseDataModel):
    id: int | None = None
    recommendation_id: str
    symbol: str
    entry_date: date
    entry_price: float
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    pnl_percentage: float | None = None
    pnl_rupees: float | None = None
    max_adverse_excursion_pct: float | None = None
    max_favorable_excursion_pct: float | None = None
    holding_sessions: int = 0
    status: TradeStatus = Field(default=TradeStatus.ACTIVE)
