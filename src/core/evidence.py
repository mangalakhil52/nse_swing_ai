"""
Evidence Graph & Lineage Tracking Module.
Maintains full empirical traceability from raw source metrics to agent findings to final CIO decisions.
Ensures zero-hallucination policy and auditable trade rationales.
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from src.core.models import EvidenceItem


class EvidenceNode(BaseModel):
    id: str
    symbol: str
    agent_name: str
    claim_type: str  # 'METRIC', 'PATTERN', 'EVENT', 'RATIO', 'FLOW'
    raw_metric: str
    observed_value: Any
    unit: str
    source: str
    timestamp: datetime | str
    verification_status: str = "VERIFIED"
    citation_url: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceGraph:
    """Directed Evidence Graph mapping conclusions to primary factual nodes."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id
        self._nodes: dict[str, list[EvidenceNode]] = {}

    def add_evidence(
        self,
        symbol: str,
        agent_name: str,
        claim_type: str,
        raw_metric: str,
        observed_value: Any,
        unit: str,
        source: str,
        timestamp: datetime | str,
        verification_status: str = "VERIFIED",
        citation_url: str | None = None,
    ) -> EvidenceNode:
        """Adds a verified evidence node to the graph."""
        node_id = f"{symbol}-{agent_name}-{raw_metric}-{int(datetime.utcnow().timestamp() * 1000)}"
        node = EvidenceNode(
            id=node_id,
            symbol=symbol,
            agent_name=agent_name,
            claim_type=claim_type,
            raw_metric=raw_metric,
            observed_value=observed_value,
            unit=unit,
            source=source,
            timestamp=timestamp,
            verification_status=verification_status,
            citation_url=citation_url,
        )

        if symbol not in self._nodes:
            self._nodes[symbol] = []
        self._nodes[symbol].append(node)
        return node

    def get_symbol_evidence(self, symbol: str) -> list[EvidenceNode]:
        """Returns all evidence nodes collected for a specific symbol."""
        return self._nodes.get(symbol, [])

    def to_evidence_items(self, symbol: str) -> list[EvidenceItem]:
        """Converts internal evidence nodes into serialized EvidenceItem contracts."""
        nodes = self.get_symbol_evidence(symbol)
        return [
            EvidenceItem(
                metric_name=node.raw_metric,
                observed_value=node.observed_value,
                unit=node.unit,
                source=node.source,
                timestamp=node.timestamp,
                verification_status=node.verification_status,
                citation_url=node.citation_url,
            )
            for node in nodes
        ]

    def verify_all_claims(self, symbol: str) -> tuple[bool, list[str]]:
        """Verifies that all registered claims for a symbol have verified data sources."""
        nodes = self.get_symbol_evidence(symbol)
        unverified: list[str] = []
        for node in nodes:
            if node.verification_status != "VERIFIED" or not node.source:
                unverified.append(f"{node.agent_name}: {node.raw_metric} (source: {node.source})")
        return (len(unverified) == 0, unverified)
