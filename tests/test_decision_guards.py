from src.quant.decision_engine import decide
from src.quant.experiment_guard import evaluate_promotion


def test_decision_fails_closed_on_missing_probability():
    result = decide(2, None, None, None, 0.1, True, True)
    assert result.action == "NO_TRADE"


def test_decision_requires_all_gates():
    result = decide(2, 0.70, 2.0, 0.55, 0.1, True, True)
    assert result.action == "TRADE"


def test_promotion_guard_rejects_weak_oos_result():
    result = evaluate_promotion(0.2, 0.51, -0.10, 6, 0.02)
    assert not result.approved


def test_promotion_guard_accepts_strong_oos_result():
    result = evaluate_promotion(1.0, 0.58, -0.20, 6, 0.10)
    assert result.approved
