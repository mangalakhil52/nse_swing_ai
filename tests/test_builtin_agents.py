from src.agents.builtin import FundamentalAgent, IPORadarAgent, NewsAgent


def test_missing_fundamentals_fail_closed():
    result = FundamentalAgent().evaluate("TRENT", {"as_of": "2026-08-24"})
    assert result.decision == "WAIT"


def test_news_uses_supplied_pit_context_only():
    result = NewsAgent().evaluate("TRENT", {"as_of": "2026-08-24", "news": {"score": 20, "headline": "x"}})
    assert result.decision == "PASS"
    assert result.evidence[0].source == "news"


def test_recent_listing_pathway():
    result = IPORadarAgent().evaluate("NEWCO", {"as_of": "2026-08-24", "listing": {"listing_date": "2026-07-01", "listing_age_days": 54}})
    assert result.decision == "PASS"
    assert result.score == 100
