"""#14L live/historical Indian equity universe normalization.

The universe provider is intentionally source-agnostic: adapters can supply NSE
(or another approved source) symbols without hardcoding securities in analysis.
"""
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class UniverseSymbol:
    symbol: str
    exchange: str = "NSE"
    listing_date: date | None = None
    delisting_date: date | None = None
    active: bool = True


class MarketUniverseService:
    @staticmethod
    def normalize(raw_symbols, as_of_date: date | None = None) -> list[UniverseSymbol]:
        as_of_date = as_of_date or date.today()
        result = []
        seen = set()
        for item in raw_symbols or []:
            if isinstance(item, str):
                symbol = item.strip().upper()
                item = UniverseSymbol(symbol=symbol)
            elif isinstance(item, UniverseSymbol):
                symbol = item.symbol.strip().upper()
                item = UniverseSymbol(symbol=symbol, exchange=item.exchange,
                                      listing_date=item.listing_date, delisting_date=item.delisting_date,
                                      active=item.active)
            else:
                symbol = str(item.get("symbol", "")).strip().upper()
                item = UniverseSymbol(symbol=symbol, exchange=item.get("exchange", "NSE"),
                                      listing_date=item.get("listing_date"), delisting_date=item.get("delisting_date"),
                                      active=item.get("active", True))
            if not symbol or symbol in seen or item.exchange.upper() != "NSE":
                continue
            if item.listing_date and item.listing_date > as_of_date:
                continue
            if item.delisting_date and item.delisting_date <= as_of_date:
                continue
            if not item.active:
                continue
            seen.add(symbol)
            result.append(item)
        return sorted(result, key=lambda x: x.symbol)
