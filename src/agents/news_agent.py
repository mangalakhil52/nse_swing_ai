"""News/Event Intelligence Specialist Agent — P0 #14E.

Consumes only point-in-time verified news and corporate-event evidence.  This
agent emits specialist evidence; it does not make trade or portfolio decisions.
"""
from datetime import date, datetime
from typing import Any
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.core.evidence import EvidenceGraph
from src.core.models import AgentOutput, CorporateAnnouncement, NewsArticle, SymbolMetadata
from src.core.types import AgentStatus, DataFreshness, SentimentType, SignalType
from src.architecture.contracts import AgentAnalysisResult, StructuredEvidence
from src.data.point_in_time import PointInTimeFilter


class NewsIntelligenceAgent(BaseAgent):
    """Specialist news/event agent with explicit PIT-safe contract output."""

    def __init__(self):
        super().__init__(agent_name="news_intelligence_agent")

    async def _analyze(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        evidence_graph: EvidenceGraph,
        run_id: str,
        context: dict[str, Any],
    ) -> AgentOutput:
        symbol = symbol_meta.symbol
        raw_articles: list[NewsArticle] = context.get("news_articles", [])
        raw_announcements: list[CorporateAnnouncement] = context.get("announcements", [])
        as_of = context.get("as_of_datetime") or context.get("as_of_date")

        if as_of is not None:
            articles = PointInTimeFilter.filter_news(raw_articles, as_of)
            as_of_date = as_of.date() if isinstance(as_of, datetime) else pd.to_datetime(as_of).date()
            announcements = PointInTimeFilter.filter_events(raw_announcements, as_of_date)
        else:
            # Without an explicit decision time, historical/live semantics are
            # ambiguous. Fail closed rather than consuming unbounded news.
            articles = []
            announcements = []

        if not articles and not announcements:
            return AgentOutput(
                agent_name=self.agent_name,
                symbol=symbol,
                run_id=run_id,
                status=AgentStatus.DATA_UNAVAILABLE,
                signal=SignalType.NEUTRAL,
                score=0.0,
                confidence=None,
                data_freshness=DataFreshness.UNKNOWN,
                metrics={"articles_found": 0, "announcements_found": 0, "status": "NEWS_UNAVAILABLE_OR_PIT_UNVERIFIED"},
                risks_identified=["No point-in-time verified news/event evidence available."],
                evidence=[],
            )

        pos_count = sum(1 for a in articles if a.sentiment == SentimentType.POSITIVE)
        neg_count = sum(1 for a in articles if a.sentiment == SentimentType.NEGATIVE)
        already_priced = sum(1 for a in articles if a.sentiment == SentimentType.ALREADY_PRICED)
        avg_materiality = round(sum(a.materiality_score for a in articles) / len(articles), 2) if articles else 0.5
        surprise_value = context.get("earnings_surprise_pct")
        surprise_pct = float(surprise_value) if surprise_value is not None else 0.0
        unpriced_catalyst_count = sum(1 for a in articles if a.is_catalyst and a.sentiment != SentimentType.ALREADY_PRICED)

        score = 65.0
        if surprise_pct > 10.0:
            score += 15.0
        elif surprise_pct > 0.0:
            score += 8.0
        elif surprise_pct < -10.0:
            score -= 25.0
        if unpriced_catalyst_count > 0:
            score += min(15.0, unpriced_catalyst_count * 7.5 * avg_materiality)
        if already_priced > 0:
            score -= min(10.0, already_priced * 4.0)

        if pos_count > neg_count:
            score += 10.0
            signal = SignalType.BULLISH
        elif neg_count > pos_count:
            score -= 20.0
            signal = SignalType.BEARISH
        else:
            signal = SignalType.NEUTRAL
        score = min(100.0, max(0.0, score))

        for article in articles[:3]:
            pub = getattr(article, "available_at", None) or getattr(article, "published_at", None)
            if pub is None:
                continue
            evidence_graph.add_evidence(
                symbol=symbol,
                agent_name=self.agent_name,
                claim_type="VERIFIED_NEWS_MATERIALITY",
                raw_metric="materiality_score",
                observed_value=f"[{article.publisher}] {article.headline} (Materiality: {article.materiality_score:.2f})",
                unit="materiality_index",
                source=article.publisher,
                timestamp=pub.isoformat() if hasattr(pub, "isoformat") else str(pub),
                citation_url=article.source_url,
            )

        risks: list[str] = []
        if neg_count > 0:
            risks.append(f"Identified {neg_count} negative press/filing headlines in the verified window.")
        if surprise_pct < -5.0:
            risks.append(f"Earnings miss detected: {abs(surprise_pct):.1f}% below reference consensus.")

        total_items = len(articles) + len(announcements)
        confidence = round(min(0.95, max(0.50, 0.60 + 0.05 * total_items + 0.10 * avg_materiality)), 2)
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
                "positive_articles": pos_count,
                "negative_articles": neg_count,
                "total_announcements": len(announcements),
                "avg_materiality": avg_materiality,
                "earnings_surprise_pct": surprise_pct,
                "unpriced_catalyst_count": unpriced_catalyst_count,
            },
            evidence=evidence_graph.to_evidence_items(symbol),
            risks_identified=risks,
        )

    async def analyze_contract(
        self,
        symbol_meta: SymbolMetadata,
        df: pd.DataFrame,
        decision_time: datetime | date,
        run_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> AgentAnalysisResult:
        """Emit the #14A AgentAnalysisResult contract with hard PIT semantics."""
        context = dict(context or {})
        context["as_of_datetime"] = decision_time
        graph = EvidenceGraph()
        output = await self._analyze(symbol_meta, df, graph, run_id, context)

        # Contract safety is based on verified records actually visible at the
        # decision time, not merely on the presence of a context key.
        raw_articles = context.get("news_articles", [])
        raw_events = context.get("announcements", [])
        pit_articles = PointInTimeFilter.filter_news(raw_articles, decision_time) if raw_articles else []
        decision_date = decision_time.date() if isinstance(decision_time, datetime) else decision_time
        pit_events = PointInTimeFilter.filter_events(raw_events, decision_date) if raw_events else []
        pit_safe = bool(pit_articles or pit_events) and output.status == AgentStatus.SUCCESS
        signal = output.signal if pit_safe else SignalType.UNKNOWN
        structured = [
            StructuredEvidence(
                source="NEWS",
                observation=item.observed_value,
                as_of=decision_time,
                direction=signal,
                strength="HIGH" if output.score >= 75 else ("MEDIUM" if output.score >= 55 else "LOW"),
                reliability=1.0 if pit_safe else 0.0,
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
            confidence=output.confidence if pit_safe and output.confidence is not None else 0.0,
            evidence=structured,
            risks=output.risks_identified,
            catalysts=["UNPRICED_NEWS_CATALYST"] if pit_safe and output.metrics.get("unpriced_catalyst_count", 0) else [],
            data_quality=None,
            pit_safe=pit_safe,
            status=output.status if pit_safe else AgentStatus.DATA_UNAVAILABLE,
            reasons=output.risks_identified if pit_safe else ["NEWS_UNAVAILABLE_OR_PIT_UNVERIFIED"],
        )
