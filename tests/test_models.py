"""
Unit tests for core Pydantic data models, EvidenceGraph, and JSON contracts.
"""

from datetime import date, datetime
import pytest

from src.core.evidence import EvidenceGraph
from src.core.models import (
    AgentOutput,
    CandidateScore,
    EvidenceItem,
    LiveQuote,
    OHLCVCandle,
    SymbolMetadata,
    TradeLevels,
    TradeRecommendation,
)
from src.core.types import (
    AgentStatus,
    ConfluenceState,
    ConvictionGrade,
    DataFreshness,
    SignalType,
    TradeStatus,
)


def test_symbol_metadata_creation():
    sym = SymbolMetadata(
        symbol="TRENT",
        company_name="Trent Limited",
        isin="INE849A01020",
        exchange="NSE",
        sector="Retail",
        industry="Apparel Retail",
        is_fno_eligible=True,
    )
    assert sym.symbol == "TRENT"
    assert sym.is_fno_eligible is True
    assert sym.asm_gsm_stage == 0


def test_ohlcv_candle_validation():
    candle = OHLCVCandle(
        timestamp=datetime(2026, 8, 14, 15, 30),
        symbol="RELIANCE",
        open=2950.0,
        high=2980.0,
        low=2940.0,
        close=2975.0,
        volume=4500000,
        delivery_volume=2800000,
        delivery_pct=62.2,
        vwap=2965.4,
        turnover_crores=1334.4,
    )
    assert candle.close == 2975.0
    assert candle.delivery_pct == 62.2


def test_evidence_graph_and_lineage():
    graph = EvidenceGraph(run_id="RUN-20260816-01")
    node = graph.add_evidence(
        symbol="TRENT",
        agent_name="technical_analysis_agent",
        claim_type="PATTERN",
        raw_metric="vcp_contraction_ratio",
        observed_value=2.45,
        unit="ratio",
        source="NSE_BHAVCOPY_EOD",
        timestamp=datetime.utcnow(),
    )
    assert node.symbol == "TRENT"
    assert node.verification_status == "VERIFIED"

    items = graph.to_evidence_items("TRENT")
    assert len(items) == 1
    assert items[0].metric_name == "vcp_contraction_ratio"

    is_verified, unverified = graph.verify_all_claims("TRENT")
    assert is_verified is True
    assert len(unverified) == 0


def test_agent_output_json_serialization():
    output = AgentOutput(
        agent_name="fundamental_agent",
        symbol="TRENT",
        run_id="RUN-01",
        status=AgentStatus.SUCCESS,
        signal=SignalType.BULLISH,
        score=85.0,
        confidence=0.90,
        data_freshness=DataFreshness.RECENT,
        metrics={"sales_growth_yoy": 24.5, "roe": 22.0},
        risks_identified=["High historical valuation P/E"],
    )
    dumped = output.model_dump()
    assert dumped["score"] == 85.0
    assert dumped["status"] == "SUCCESS"
    assert dumped["signal"] == "BULLISH"


def test_trade_recommendation_model():
    levels = TradeLevels(
        symbol="TRENT",
        current_market_price=7150.0,
        entry_trigger_price=7160.0,
        stop_loss_price=6900.0,
        risk_rupees=260.0,
        risk_percentage=3.63,
        target_1=7680.0,
        target_2=8000.0,
        target_3=8450.0,
        risk_reward_t1=2.0,
        risk_reward_t2=3.23,
        risk_reward_t3=4.96,
        position_size_shares=38,
        allocated_capital_rupees=272080.0,
        invalidation_criteria="Daily close below 20 EMA at 6900",
    )
    rec = TradeRecommendation(
        recommendation_id="REC-20260816-TRENT",
        run_id="RUN-01",
        symbol="TRENT",
        company_name="Trent Limited",
        sector="Retail",
        recommendation_date=date(2026, 8, 16),
        conviction=ConvictionGrade.A_PLUS,
        composite_score=89.5,
        levels=levels,
        technical_setup_description="VCP Breakout with 2.4x volume surge",
        catalyst_summary="Rapid store expansion and strong Q1 results",
        fundamental_summary="Sales YoY +28%, PAT YoY +34%, ROE 22%",
        sector_context="Retail sector in top 2 momentum rank",
        market_regime="STRONG_BULL",
        major_risks=["Broader market pullback"],
        invalidation_rules="Close below 6900",
        why_this_trade=["1. Structural tight base", "2. Strong RS vs Nifty", "3. Institutional volume"],
        evidence_dossier=[],
        status=TradeStatus.PENDING_ENTRY,
    )
    assert rec.conviction == ConvictionGrade.A_PLUS
    assert rec.levels.risk_reward_t1 == 2.0
