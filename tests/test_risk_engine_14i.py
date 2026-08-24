"""#14I independent risk engine tests."""
from datetime import date
from src.architecture.contracts import VetoType
from src.architecture.risk_engine import RiskSynthesisService

D = date(2026, 6, 30)

def test_valid_risk_returns_position_ceiling():
    out = RiskSynthesisService.evaluate("TRENT", D, entry_price=100, stop_loss=95,
        risk_budget_pct=1.0, max_position_size_pct=10.0, regime_risk_multiplier=1.0)
    assert out.passed_risk_veto is True
    assert out.stop_loss_valid is True
    assert out.max_position_size_pct == 10.0


def test_stop_above_entry_is_vetoed():
    out = RiskSynthesisService.evaluate("TRENT", D, entry_price=100, stop_loss=105)
    assert out.passed_risk_veto is False
    assert out.veto_category == VetoType.RISK_VETO


def test_missing_prices_are_vetoed():
    out = RiskSynthesisService.evaluate("TRENT", D, entry_price=None, stop_loss=95)
    assert out.passed_risk_veto is False
    assert out.max_position_size_pct == 0.0


def test_regime_multiplier_reduces_position_ceiling():
    normal = RiskSynthesisService.evaluate("TRENT", D, entry_price=100, stop_loss=95,
        risk_budget_pct=1.0, max_position_size_pct=20, regime_risk_multiplier=1.0)
    reduced = RiskSynthesisService.evaluate("TRENT", D, entry_price=100, stop_loss=95,
        risk_budget_pct=1.0, max_position_size_pct=20, regime_risk_multiplier=.5)
    assert reduced.max_position_size_pct < normal.max_position_size_pct


def test_invalid_risk_budget_is_vetoed():
    out = RiskSynthesisService.evaluate("TRENT", D, entry_price=100, stop_loss=95, risk_budget_pct=0)
    assert out.passed_risk_veto is False
    assert out.veto_category == VetoType.RISK_VETO


def test_position_is_always_capped():
    out = RiskSynthesisService.evaluate("TRENT", D, entry_price=100, stop_loss=99.9,
        risk_budget_pct=10, max_position_size_pct=7.5, regime_risk_multiplier=2)
    assert out.max_position_size_pct <= 7.5
