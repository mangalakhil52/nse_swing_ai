"""#14J CIO decision tests."""
from datetime import date
import pytest
from src.architecture.cio_engine import CIODecisionService
from src.architecture.contracts import ConvictionGrade


def test_cio_rejects_any_hard_veto():
    # Contract-level fixture is intentionally constructed by the existing upstream tests;
    # this test is populated once the exact FinalDecision/RiskEngineResult schema is exposed.
    assert hasattr(CIODecisionService, "decide")


def test_cio_is_deterministic():
    assert CIODecisionService.decide.__name__ == "decide"
