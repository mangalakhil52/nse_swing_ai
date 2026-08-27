import pytest

from src.candidate_discovery import CandidateDiscoveryEngine


def test_historical_discovery_never_falls_back_to_current_universe():
    with pytest.raises(TypeError, match="explicit SymbolMetadata"):
        CandidateDiscoveryEngine.discover_candidates(
            universe=None,
            as_of_date="2024-01-10",
            market_data_map={},
            mode="HISTORICAL",
        )
