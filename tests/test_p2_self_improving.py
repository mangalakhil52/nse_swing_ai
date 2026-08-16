"""
Unit tests for P2 Self-Improving Engine Upgrades:
  1. Agent Performance Attribution (Desk Win Rate, Alpha, Brier Score)
  2. Feature Attribution (Shapley-style Factor Correlation with Returns)
  3. Walk-Forward Model Selection & Optimization (IS/OOS Efficiency)
  4. Model Versioning & Champion/Challenger Promotion Engine
  5. Automatic Strategy Degradation Monitoring & Risk Guard
"""

from pathlib import Path
import pytest

from src.backtest.walk_forward import WalkForwardOptimizer
from src.database.model_registry import ModelRegistry
from src.shadow.attribution import AgentAttributionEngine
from src.shadow.degradation_monitor import StrategyDegradationMonitor


def test_agent_performance_attribution():
    history = [
        {"pnl_pct": 8.5, "agent_scores": {"technical_agent": 85.0, "rs_agent": 90.0}},
        {"pnl_pct": 12.0, "agent_scores": {"technical_agent": 88.0, "rs_agent": 92.0}},
        {"pnl_pct": -4.0, "agent_scores": {"technical_agent": 60.0, "rs_agent": 55.0}},
    ]
    attrib = AgentAttributionEngine.evaluate_desk_attribution(history)

    assert "technical_agent" in attrib
    assert attrib["technical_agent"].total_recommendations == 3
    assert attrib["technical_agent"].winning_trades == 2
    assert attrib["technical_agent"].win_rate_pct == 66.7
    assert attrib["technical_agent"].brier_score >= 0.0


def test_feature_attribution():
    history = [
        {"pnl_pct": 10.0, "features": {"mansfield_rs": 15.0, "vcp_contraction_ratio": 0.85}},
        {"pnl_pct": 6.0, "features": {"mansfield_rs": 8.0, "vcp_contraction_ratio": 0.70}},
        {"pnl_pct": -3.0, "features": {"mansfield_rs": -2.0, "vcp_contraction_ratio": 0.40}},
    ]
    features = AgentAttributionEngine.evaluate_feature_attribution(history)

    assert len(features) > 0
    assert features[0].importance_rank == 1
    assert features[0].correlation_with_return != 0.0


def test_walk_forward_optimization():
    results = WalkForwardOptimizer.run_walk_forward_optimization([], num_windows=3)

    assert len(results) == 3
    assert results[0].efficiency_ratio >= 0.70
    assert "technical_weight" in results[0].optimal_weights


def test_model_registry_champion_challenger(tmp_path: Path):
    reg_path = tmp_path / "model_registry.json"
    registry = ModelRegistry(registry_file=reg_path)

    champ = registry.get_champion_model()
    assert champ.role == "CHAMPION"
    assert champ.version == "v1.2.0-champion"

    promoted = registry.evaluate_challenger_promotion()
    assert promoted is True

    new_champ = registry.get_champion_model()
    assert new_champ.version == "v1.3.0-challenger"


def test_strategy_degradation_monitor():
    # Healthy trades test
    healthy_trades = [{"pnl_pct": 5.0 + (i % 3)} for i in range(10)]
    h_status = StrategyDegradationMonitor.evaluate_strategy_health(healthy_trades)
    assert h_status.is_degraded is False
    assert h_status.recommended_risk_multiplier == 1.0

    # Degraded trades test (consecutive losses & negative PnLs)
    degraded_trades = [{"pnl_pct": -4.0} for _ in range(8)]
    d_status = StrategyDegradationMonitor.evaluate_strategy_health(degraded_trades)
    assert d_status.is_degraded is True
    assert d_status.recommended_risk_multiplier == 0.50
    assert d_status.recommended_stance == "DEFENSIVE"
