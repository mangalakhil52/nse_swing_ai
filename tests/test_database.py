"""
Unit tests for database persistence and repository operations.
"""

from datetime import date, datetime
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.models import (
    AgentOutput,
    CandidateScore,
    EvidenceItem,
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
from src.database.repository import DatabaseRepository
from src.database.schema import Base


@pytest.fixture
def in_memory_repo():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    repo = DatabaseRepository(session)
    yield repo
    session.rollback()
    session.close()


def test_upsert_and_retrieve_securities(in_memory_repo: DatabaseRepository):
    securities = [
        SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail", is_fno_eligible=True),
        SymbolMetadata(symbol="BEL", company_name="Bharat Electronics", sector="Defence", is_fno_eligible=True),
    ]
    count = in_memory_repo.upsert_securities(securities)
    assert count == 2

    active = in_memory_repo.get_all_active_securities()
    assert len(active) == 2
    trent = in_memory_repo.get_security_by_symbol("TRENT")
    assert trent is not None
    assert trent.company_name == "Trent Ltd"


def test_save_and_retrieve_ohlcv(in_memory_repo: DatabaseRepository):
    securities = [SymbolMetadata(symbol="TRENT", company_name="Trent Ltd")]
    in_memory_repo.upsert_securities(securities)
    trent = in_memory_repo.get_security_by_symbol("TRENT")

    df = pd.DataFrame([
        {
            "timestamp": datetime(2026, 8, 14, 15, 30),
            "open": 7000.0,
            "high": 7200.0,
            "low": 6950.0,
            "close": 7180.0,
            "volume": 1500000,
            "delivery_volume": 900000,
            "delivery_pct": 60.0,
            "vwap": 7120.0,
            "turnover_crores": 107.0,
        }
    ])
    saved = in_memory_repo.save_ohlcv_bars(trent.id, df)
    assert saved == 1

    retrieved = in_memory_repo.get_ohlcv_dataframe("TRENT", lookback_days=10)
    assert len(retrieved) == 1
    assert retrieved.iloc[0]["close"] == 7180.0


def test_save_agent_run_and_recommendation(in_memory_repo: DatabaseRepository):
    sec = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd")
    in_memory_repo.upsert_securities([sec])

    run = in_memory_repo.create_agent_run(
        run_id="RUN-TEST-01",
        market_regime="STRONG_BULL",
        universe_size=2000,
        quant_candidates_count=50,
    )
    assert run.id == "RUN-TEST-01"

    output = AgentOutput(
        agent_name="technical_agent",
        symbol="TRENT",
        run_id="RUN-TEST-01",
        status=AgentStatus.SUCCESS,
        signal=SignalType.BULLISH,
        score=90.0,
        confidence=0.95,
        metrics={"pattern": "VCP"},
    )
    in_memory_repo.save_agent_output(output)

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
        invalidation_criteria="Daily close below 6900",
    )
    rec = TradeRecommendation(
        recommendation_id="REC-TEST-01",
        run_id="RUN-TEST-01",
        symbol="TRENT",
        company_name="Trent Ltd",
        sector="Retail",
        recommendation_date=date.today(),
        conviction=ConvictionGrade.A_PLUS,
        composite_score=91.0,
        levels=levels,
        technical_setup_description="VCP Breakout",
        catalyst_summary="Strong earnings",
        fundamental_summary="High growth",
        sector_context="Sector leader",
        market_regime="STRONG_BULL",
        major_risks=[],
        invalidation_rules="Close below 6900",
        why_this_trade=["High momentum"],
        evidence_dossier=[],
        status=TradeStatus.PENDING_ENTRY,
    )
    in_memory_repo.save_trade_recommendation(rec)
    in_memory_repo.complete_agent_run("RUN-TEST-01", status="COMPLETED", recommended_count=1)
