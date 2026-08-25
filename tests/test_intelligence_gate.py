from types import SimpleNamespace

from src.quant.intelligence_gate import IntelligenceGateConfig, select_normal, select_recent


def _candidate(symbol, trend=80.0, rs=5.0):
    return SimpleNamespace(symbol=symbol, trend_score=trend, mansfield_rs=rs)


def test_normal_gate_requires_bullish_technical_and_positive_quality():
    rows = [
        SimpleNamespace(symbol="GOOD", signal="BULLISH", technical_score=82.0, pit_safe=True),
        SimpleNamespace(symbol="NEUTRAL", signal="NEUTRAL", technical_score=95.0, pit_safe=True),
        SimpleNamespace(symbol="WEAK", signal="BULLISH", technical_score=74.9, pit_safe=True),
        SimpleNamespace(symbol="PITBAD", signal="BULLISH", technical_score=90.0, pit_safe=False),
    ]
    candidates = {
        "GOOD": _candidate("GOOD", 80, 5),
        "NEUTRAL": _candidate("NEUTRAL", 100, 10),
        "WEAK": _candidate("WEAK", 100, 10),
        "PITBAD": _candidate("PITBAD", 100, 10),
    }
    out = select_normal(rows, candidates)
    assert [x.symbol for x in out] == ["GOOD"]


def test_normal_gate_caps_and_orders_by_technical_then_rs():
    rows = [SimpleNamespace(symbol=f"S{i}", signal="BULLISH", technical_score=80+i, pit_safe=True) for i in range(4)]
    candidates = {f"S{i}": _candidate(f"S{i}", 80, i) for i in range(4)}
    out = select_normal(rows, candidates, IntelligenceGateConfig(normal_max_candidates=2))
    assert [x.symbol for x in out] == ["S3", "S2"]


def test_recent_gate_is_separate_from_normal_track():
    rows = [
        SimpleNamespace(symbol="NEW1", score=91.0, median_turnover_crores=10),
        SimpleNamespace(symbol="NEW2", score=74.9, median_turnover_crores=100),
        SimpleNamespace(symbol="NEW3", score=88.0, median_turnover_crores=5),
    ]
    candidates = {x.symbol: _candidate(x.symbol) for x in rows}
    out = select_recent(rows, candidates, IntelligenceGateConfig(recent_max_candidates=2))
    assert [x.symbol for x in out] == ["NEW1", "NEW3"]
