"""Stable contracts for specialist intelligence agents."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentEvidence:
    key: str
    value: Any
    source: str
    as_of: str
    impact: str = "NEUTRAL"


@dataclass(frozen=True)
class AgentResult:
    agent: str
    symbol: str
    status: str
    score: float | None = None
    confidence: float | None = None
    decision: str = "WAIT"
    evidence: tuple[AgentEvidence, ...] = field(default_factory=tuple)
    risks: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def telemetry(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "agent"
        payload["evidence"] = [asdict(item) for item in self.evidence]
        return payload


class Agent:
    """Minimal specialist-agent protocol.

    Implementations must be deterministic for the same point-in-time inputs and
    must return evidence with an explicit source/as-of timestamp.
    """

    name = "BASE"

    def evaluate(self, symbol: str, context: Mapping[str, Any]) -> AgentResult:
        raise NotImplementedError
