"""#14L live/historical Indian equity universe normalization."""
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
        for raw_item in raw_symbols or []:
            if raw_item is None:
                continue
            if isinstance(raw_item, str):
                item = UniverseSymbol(symbol=raw_item.strip().upper())
            elif isinstance(raw_item, UniverseSymbol):
                item = UniverseSymbol(
                    symbol=raw_item.symbol.strip().upper(), exchange=raw_item.exchange,
                    listing_date=raw_item.listing_date, delisting_date=raw_item.delisting_date,
                    active=raw_item.active,
                )
            elif isinstance(raw_item, dict):
                item = UniverseSymbol(
                    symbol=str(raw_item.get("symbol", "")).strip().upper(),
                    exchange=str(raw_item.get("exchange", "NSE")),
                    listing_date=raw_item.get("listing_date"),
                    delisting_date=raw_item.get("delisting_date"),
                    active=bool(raw_item.get("active", True)),
                )
            else:
                continue

            symbol = item.symbol
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
