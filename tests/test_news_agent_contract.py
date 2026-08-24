"""P0 #14E — News/Event specialist contract and PIT integration tests."""
import asyncio
from datetime import datetime
import pandas as pd

from src.agents.news_agent import NewsIntelligenceAgent
from src.architecture.contracts import AgentAnalysisResult
from src.core.evidence import EvidenceGraph
from src.core.models import NewsArticle, SymbolMetadata
from src.core.types import AgentStatus, SentimentType, SignalType, SourceTier


def _article(published_at, sentiment=SentimentType.POSITIVE, materiality=0.8):
    return NewsArticle(
        symbol="SYNTH",
        headline="Verified corporate development",
        publisher="Reuters",
        source_tier=SourceTier.TIER_1,
        published_at=published_at,
        sentiment=sentiment,
        materiality_score=materiality,
    )


def test_news_contract_requires_explicit_decision_time():
    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="SYNTH", company_name="Synthetic Ltd")
    decision_time = datetime(2026, 8, 19, 10, 0)
    result = asyncio.run(agent.analyze_contract(
        meta, pd.DataFrame(), decision_time,
        context={"news_articles": [_article(datetime(2026, 8, 18, 12, 0))]},
    ))
    assert isinstance(result, AgentAnalysisResult)
    assert result.pit_safe is True
    assert result.signal == SignalType.BULLISH
    assert result.status == AgentStatus.SUCCESS


def test_news_contract_missing_data_fails_closed():
    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="SYNTH", company_name="Synthetic Ltd")
    result = asyncio.run(agent.analyze_contract(
        meta, pd.DataFrame(), datetime(2026, 8, 19, 10, 0), context={"news_articles": []}
    ))
    assert result.pit_safe is False
    assert result.signal == SignalType.UNKNOWN
    assert result.score == 0.0
    assert result.status == AgentStatus.DATA_UNAVAILABLE


def test_news_contract_excludes_future_article():
    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="SYNTH", company_name="Synthetic Ltd")
    decision_time = datetime(2026, 8, 19, 10, 0)
    future = _article(datetime(2026, 8, 20, 10, 0), SentimentType.NEGATIVE, 1.0)
    result = asyncio.run(agent.analyze_contract(meta, pd.DataFrame(), decision_time,
                                                context={"news_articles": [future]}))
    assert result.pit_safe is False
    assert result.signal == SignalType.UNKNOWN
    assert result.score == 0.0


def test_future_news_mutation_cannot_change_contract_result():
    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="SYNTH", company_name="Synthetic Ltd")
    decision_time = datetime(2026, 8, 19, 10, 0)
    past = _article(datetime(2026, 8, 18, 12, 0), SentimentType.POSITIVE, 0.7)
    future_a = _article(datetime(2026, 8, 20, 10, 0), SentimentType.NEUTRAL, 0.2)
    future_b = _article(datetime(2026, 8, 20, 10, 0), SentimentType.NEGATIVE, 1.0)
    r1 = asyncio.run(agent.analyze_contract(meta, pd.DataFrame(), decision_time,
                                            context={"news_articles": [past, future_a]}))
    r2 = asyncio.run(agent.analyze_contract(meta, pd.DataFrame(), decision_time,
                                            context={"news_articles": [past, future_b]}))
    assert r1.model_dump() == r2.model_dump()


def test_no_implicit_earnings_surprise_default():
    agent = NewsIntelligenceAgent()
    meta = SymbolMetadata(symbol="SYNTH", company_name="Synthetic Ltd")
    decision_time = datetime(2026, 8, 19, 10, 0)
    article = _article(datetime(2026, 8, 18, 12, 0), SentimentType.NEUTRAL)
    result = asyncio.run(agent._analyze(meta, pd.DataFrame(), EvidenceGraph(), "run",
                                        {"news_articles": [article], "as_of_datetime": decision_time}))
    assert result.metrics["earnings_surprise_pct"] == 0.0
