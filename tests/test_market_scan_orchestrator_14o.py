"""#14O whole-market scan tests."""
from datetime import date
from types import SimpleNamespace
from src.architecture.market_scan_orchestrator import MarketScanOrchestrator


class Universe:
    def source(self, _):
        return ["AAA", "BBB", "CCC"]
    def normalize(self, raw, _):
        return [SimpleNamespace(symbol=x) for x in raw]


class Data:
    def fetch(self, symbol, _):
        return SimpleNamespace(frame={"symbol": symbol})


def test_filters_before_decision_and_isolates_failures():
    seen = []
    def discover(symbol, frame, _):
        if symbol == "CCC":
            raise RuntimeError("bad data")
        return SimpleNamespace(eligible=symbol == "AAA")
    def decision(symbol, frame, _):
        seen.append(symbol.symbol)
        return {"action": "BUY"}
    out = MarketScanOrchestrator(Universe(), Data(), discover, decision, max_workers=2).scan(date(2026,6,30))
    assert seen == ["AAA"]
    assert [x.symbol for x in out.decisions] == ["AAA"]
    assert any(x.symbol == "BBB" and x.status == "FILTERED" for x in out.items)
    assert any(x.symbol == "CCC" and x.status == "FAILED" for x in out.items)


def test_results_are_sorted_deterministically():
    universe = Universe()
    universe.source = lambda _: ["ZZZ", "AAA"]
    def discover(symbol, frame, _): return SimpleNamespace(eligible=False)
    out = MarketScanOrchestrator(universe, Data(), discover, lambda *x: None).scan(date(2026,6,30))
    assert [x.symbol for x in out.items] == ["AAA", "ZZZ"]
