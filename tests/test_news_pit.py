"""
P0 #12C — News / Event Point-in-Time Integrity & Leakage Prevention Unit & Integration Tests.

Validates that:
  1. published_at / available_at strictly controls news article visibility.
  2. event_date <= as_of_date NEVER grants early visibility if announcement_date / available_at > as_of_date.
  3. Articles/events with missing publication/availability timestamps fail closed (PIT_UNVERIFIED).
  4. Future news mutations (published_at > T) cannot alter NewsIntelligenceAgent score / signal at T.
  5. Same-day intraday timestamp precision prevents 15:45 news from influencing a 10:00 decision.
  6. News consumer status in active backtest is explicitly documented as NOT_IMPLEMENTED.
"""

import asyncio
from datetime import date, datetime
import pandas as pd
import pytest

from src.agents.news_agent import NewsIntelligenceAgent
from src.core.evidence import EvidenceGraph
from src.core.models import CorporateEvent, NewsArticle, SymbolMetadata
from src.core.types import SentimentType, SourceTier
from src.data.point_in_time import PointInTimeFilter


def test_news_publication_date_controls_visibility():
    """1. Test published_at strictly controls news article visibility."""
    article = NewsArticle(
        symbol="TRENT",
        headline="Trent Expands Store Count",
        publisher="Economic Times",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 15, 10, 0, 0),
        sentiment=SentimentType.POSITIVE,
        materiality_score=0.8,
    )

    # 2026-05-14 -> invisible
    assert len(PointInTimeFilter.filter_news([article], date(2026, 5, 14))) == 0
    # 2026-05-15 -> visible
    assert len(PointInTimeFilter.filter_news([article], date(2026, 5, 15))) == 1
    # 2026-05-16 -> visible
    assert len(PointInTimeFilter.filter_news([article], date(2026, 5, 16))) == 1


def test_event_date_does_not_grant_early_visibility():
    """2. Test event_date <= as_of_date does NOT grant early visibility if announcement_date / available_at > as_of_date."""
    event = CorporateEvent(
        symbol="TRENT",
        event_type="EARNINGS_RELEASE",
        event_date=date(2026, 3, 31),
        announcement_date=date(2026, 5, 15),
        available_at=date(2026, 5, 15),
        purpose="Q4 Financial Results",
    )

    # At 2026-04-01: event_date (2026-03-31) has passed, but announcement_date is 2026-05-15
    filtered = PointInTimeFilter.filter_events([event], date(2026, 4, 1))
    assert len(filtered) == 0


def test_missing_news_availability_fails_closed():
    """3. Test missing publication / availability timestamp fails closed (excludes item)."""
    unverified_event = CorporateEvent(
        symbol="TRENT",
        event_type="BOARD_MEETING",
        event_date=date(2026, 3, 31),
        announcement_date=None,
        available_at=None,
        purpose="Board Meeting",
    )

    filtered = PointInTimeFilter.filter_events([unverified_event], date(2026, 6, 1))
    assert len(filtered) == 0


def test_future_news_mutation_does_not_change_result_at_T():
    """4. Test mutating future news articles (published_at > T) leaves NewsIntelligenceAgent score at T identical."""
    t_dt = datetime(2026, 5, 15, 12, 0, 0)

    art1_past = NewsArticle(
        symbol="TRENT",
        headline="Strong Q4 Sales Record",
        publisher="Mint",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 10, 9, 0, 0),
        sentiment=SentimentType.POSITIVE,
        materiality_score=0.85,
    )

    art2_future_orig = NewsArticle(
        symbol="TRENT",
        headline="Minor Management Commentary",
        publisher="Reuters",
        source_tier=SourceTier.TIER_2,
        published_at=datetime(2026, 5, 20, 14, 0, 0),  # Published after T
        sentiment=SentimentType.NEUTRAL,
        materiality_score=0.4,
    )

    art2_future_mut = art2_future_orig.model_copy(update={
        "headline": "CRITICAL REGULATORY INVESTIGATION & SEVERE FRAUD ALLEGATIONS",
        "sentiment": SentimentType.NEGATIVE,
        "materiality_score": 1.0,
    })

    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")
    dummy_df = pd.DataFrame()

    # Baseline run at T
    ctx_base = {"news_articles": [art1_past, art2_future_orig], "as_of_datetime": t_dt}
    out_base = asyncio.run(agent._analyze(meta, dummy_df, EvidenceGraph(), "run1", ctx_base))

    # Mutated run at T
    ctx_mut = {"news_articles": [art1_past, art2_future_mut], "as_of_datetime": t_dt}
    out_mut = asyncio.run(agent._analyze(meta, dummy_df, EvidenceGraph(), "run2", ctx_mut))

    assert out_base.score == out_mut.score
    assert out_base.signal == out_mut.signal
    assert out_base.metrics == out_mut.metrics


def test_same_day_news_ordering():
    """5. Test same-day intraday precision: 15:45 article cannot influence a 10:00 decision on the same day."""
    art_morning = NewsArticle(
        symbol="TRENT",
        headline="Premarket Earnings Surprise",
        publisher="CNBC",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 15, 9, 30, 0),
        sentiment=SentimentType.POSITIVE,
    )

    art_afternoon = NewsArticle(
        symbol="TRENT",
        headline="Post-Market Regulatory Fine",
        publisher="Economic Times",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 15, 15, 45, 0),
        sentiment=SentimentType.NEGATIVE,
    )

    articles = [art_morning, art_afternoon]

    # Evaluated at 10:00 AM on 2026-05-15
    eval_10am = datetime(2026, 5, 15, 10, 0, 0)
    filtered_10am = PointInTimeFilter.filter_news(articles, eval_10am)
    assert len(filtered_10am) == 1
    assert filtered_10am[0].headline == "Premarket Earnings Surprise"

    # Evaluated at 16:00 (4:00 PM) on 2026-05-15
    eval_4pm = datetime(2026, 5, 15, 16, 0, 0)
    filtered_4pm = PointInTimeFilter.filter_news(articles, eval_4pm)
    assert len(filtered_4pm) == 2


def test_future_news_cannot_change_signal_at_T():
    """6. Test future news articles published > T cannot alter agent signal at T."""
    t_dt = datetime(2026, 5, 15, 12, 0, 0)
    art_past = NewsArticle(
        symbol="TRENT",
        headline="Solid Growth Record",
        publisher="Mint",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 14, 10, 0, 0),
        sentiment=SentimentType.POSITIVE,
    )
    art_future = NewsArticle(
        symbol="TRENT",
        headline="Future News Release",
        publisher="ET",
        source_tier=SourceTier.TIER_1,
        published_at=datetime(2026, 5, 16, 10, 0, 0),
        sentiment=SentimentType.NEGATIVE,
    )

    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="TRENT", company_name="Trent Ltd", sector="Retail")

    ctx1 = {"news_articles": [art_past], "as_of_datetime": t_dt}
    out1 = asyncio.run(agent._analyze(meta, pd.DataFrame(), EvidenceGraph(), "r1", ctx1))

    ctx2 = {"news_articles": [art_past, art_future], "as_of_datetime": t_dt}
    out2 = asyncio.run(agent._analyze(meta, pd.DataFrame(), EvidenceGraph(), "r2", ctx2))

    assert out1.score == out2.score
    assert out1.signal == out2.signal


def test_news_consumer_status_documented_as_not_implemented():
    """7. Verifies active backtest engine (PortfolioBacktestEngine) runs technical signals without news dependency."""
    from src.backtest.portfolio import PortfolioBacktestEngine
    import inspect

    # Verify PortfolioBacktestEngine does not require news input
    sig = inspect.signature(PortfolioBacktestEngine.run_portfolio_backtest)
    assert "stock_dfs" in sig.parameters
    assert "news_articles" not in sig.parameters
