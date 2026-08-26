"""Deterministic baseline specialists backed only by supplied PIT context.

These are intentionally conservative. They never invent fundamentals, news or
listing metadata. Missing context produces WAIT with an explicit reason.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping

from src.agents.contracts import Agent, AgentEvidence, AgentResult


def _as_of(context: Mapping[str, Any]) -> str:
    value = context.get("as_of") or context.get("as_of_date")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value or "unknown")


class FundamentalAgent(Agent):
    name = "FUNDAMENTAL"

    def evaluate(self, symbol: str, context: Mapping[str, Any]) -> AgentResult:
        data = context.get("fundamentals")
        if not isinstance(data, Mapping):
            return AgentResult(self.name, symbol, "WAITING", decision="WAIT", reason="No point-in-time fundamental dataset supplied")
        score = data.get("score")
        if score is None:
            return AgentResult(self.name, symbol, "WAITING", decision="WAIT", reason="Fundamental score unavailable")
        score = float(score)
        decision = "PASS" if score >= 60 else "REJECT"
        evidence = tuple(AgentEvidence(str(k), v, "fundamentals", _as_of(context)) for k, v in data.items() if k != "score")
        return AgentResult(self.name, symbol, "COMPLETE", score, min(1.0, abs(score - 50) / 50), decision, evidence, reason=f"PIT fundamental score {score:.1f}")


class NewsAgent(Agent):
    name = "NEWS INTEL"

    def evaluate(self, symbol: str, context: Mapping[str, Any]) -> AgentResult:
        data = context.get("news")
        if not isinstance(data, Mapping):
            return AgentResult(self.name, symbol, "WAITING", decision="WAIT", reason="No point-in-time news dataset supplied")
        score = data.get("score")
        if score is None:
            return AgentResult(self.name, symbol, "WAITING", decision="WAIT", reason="News impact score unavailable")
        score = float(score)
        decision = "PASS" if score > 0 else ("REJECT" if score < 0 else "WAIT")
        evidence = tuple(AgentEvidence(str(k), v, str(data.get("source", "news")), _as_of(context)) for k, v in data.items() if k not in {"score", "source"})
        return AgentResult(self.name, symbol, "COMPLETE", score, min(1.0, abs(score) / 100), decision, evidence, reason=f"PIT news impact score {score:.1f}")


class IPORadarAgent(Agent):
    name = "IPO RADAR"

    def evaluate(self, symbol: str, context: Mapping[str, Any]) -> AgentResult:
        data = context.get("listing")
        if not isinstance(data, Mapping) or data.get("listing_date") is None:
            return AgentResult(self.name, symbol, "WAITING", decision="WAIT", reason="No point-in-time listing metadata supplied")
        age = data.get("listing_age_days")
        if age is None:
            return AgentResult(self.name, symbol, "WAITING", decision="WAIT", reason="Listing age unavailable")
        age = int(age)
        eligible = 0 <= age <= int(context.get("ipo_window_days", 120))
        decision = "PASS" if eligible else "REJECT"
        evidence = (AgentEvidence("listing_age_days", age, "listing_metadata", _as_of(context)), AgentEvidence("listing_date", data.get("listing_date"), "listing_metadata", _as_of(context)))
        return AgentResult(self.name, symbol, "COMPLETE", 100.0 if eligible else 0.0, 1.0, decision, evidence, reason="Recent-listing pathway evaluated")
