"""#14I deterministic pre-trade risk validation.

Risk is independent of conviction. This layer validates risk structure and
returns a veto/position ceiling; it does not place trades or choose symbols.
"""
from src.architecture.contracts import RiskEngineResult, VetoType


class RiskSynthesisService:
    """Conservative risk gate using explicit, caller-supplied risk inputs."""

    @staticmethod
    def evaluate(symbol: str, decision_time, *, entry_price: float | None,
                 stop_loss: float | None, risk_budget_pct: float = 1.0,
                 max_position_size_pct: float = 10.0,
                 regime_risk_multiplier: float = 1.0) -> RiskEngineResult:
        reasons: list[str] = []

        if entry_price is None or stop_loss is None or entry_price <= 0 or stop_loss <= 0:
            return RiskEngineResult(symbol=symbol, decision_time=decision_time,
                passed_risk_veto=False, veto_category=VetoType.RISK_VETO,
                max_position_size_pct=0.0, stop_loss_valid=False,
                reasons=["MISSING_OR_INVALID_ENTRY_STOP"])

        stop_distance = (entry_price - stop_loss) / entry_price
        if stop_distance <= 0:
            return RiskEngineResult(symbol=symbol, decision_time=decision_time,
                passed_risk_veto=False, veto_category=VetoType.RISK_VETO,
                max_position_size_pct=0.0, stop_loss_valid=False,
                reasons=["STOP_MUST_BE_BELOW_ENTRY_FOR_LONG_RISK_MODEL"])

        if risk_budget_pct <= 0 or max_position_size_pct <= 0 or regime_risk_multiplier <= 0:
            return RiskEngineResult(symbol=symbol, decision_time=decision_time,
                passed_risk_veto=False, veto_category=VetoType.RISK_VETO,
                max_position_size_pct=0.0, stop_loss_valid=True,
                reasons=["INVALID_RISK_PARAMETERS"])

        raw_position = risk_budget_pct / (stop_distance * 100.0) * 100.0
        capped = min(max_position_size_pct, raw_position * regime_risk_multiplier)
        if capped <= 0:
            return RiskEngineResult(symbol=symbol, decision_time=decision_time,
                passed_risk_veto=False, veto_category=VetoType.RISK_VETO,
                max_position_size_pct=0.0, stop_loss_valid=True,
                reasons=["NO_PERMISSIBLE_POSITION_SIZE"])

        reasons.append(f"STOP_DISTANCE_PCT={stop_distance * 100:.2f}")
        reasons.append(f"RISK_BUDGET_PCT={risk_budget_pct:.2f}")
        reasons.append(f"REGIME_RISK_MULTIPLIER={regime_risk_multiplier:.2f}")
        return RiskEngineResult(symbol=symbol, decision_time=decision_time,
            passed_risk_veto=True, max_position_size_pct=round(capped, 4),
            stop_loss_valid=True, reasons=reasons)
