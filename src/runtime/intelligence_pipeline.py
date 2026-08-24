"""#14S bridge from the whole-market scanner to the existing CIO pipeline."""
from __future__ import annotations
import uuid
import pandas as pd
from src.agents.cio_orchestrator import CIOOrchestrator
from src.candidate_discovery import CandidateDiscoveryEngine, CandidateDiscoveryConfig
from src.core.models import SymbolMetadata


class ProductionIntelligencePipeline:
    """Uses the repository's existing candidate discovery and CIO implementation.

    Missing downstream datasets are passed through as missing; agents remain
    responsible for returning DATA_UNAVAILABLE rather than receiving synthetic data.
    """
    def __init__(self, cio: CIOOrchestrator | None = None, discovery_config: CandidateDiscoveryConfig | None = None):
        self.cio = cio or CIOOrchestrator()
        self.discovery_config = discovery_config or CandidateDiscoveryConfig()

    def discover(self, symbol: str, frame: pd.DataFrame, as_of_date):
        results = CandidateDiscoveryEngine.discover_candidates(
            universe=[symbol], as_of_date=as_of_date, market_data_map={symbol: frame},
            config=self.discovery_config, mode="LIVE",
        )
        return results[0]

    def decide(self, symbol, frame: pd.DataFrame, as_of_date, context: dict | None = None):
        symbol_name = symbol.symbol if hasattr(symbol, "symbol") else str(symbol)
        metadata = symbol if isinstance(symbol, SymbolMetadata) else SymbolMetadata(symbol=symbol_name, company_name=symbol_name)
        return self.cio.analyze_candidate(metadata, frame, run_id=str(uuid.uuid4()), context=context or {})
