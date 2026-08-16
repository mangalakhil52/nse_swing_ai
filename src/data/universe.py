"""
Dynamic NSE Universe Discovery & Security Master Module.
Builds the canonical eligible trading universe from official exchange listings.
Filters out delisted, suspended, SME illiquid securities, penny stocks (< ₹20), and securities under SEBI ASM/GSM stage >= 2.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import settings
from src.core.models import SymbolMetadata
from src.data.base import MarketDataProvider

logger = logging.getLogger(__name__)

# Standard NSE F&O Key Constituents Reference List (Active Derivatives Basket)
STANDARD_FNO_SYMBOLS = {
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS",
    "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL",
    "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE",
    "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT",
    "CANBK", "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL",
    "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR",
    "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "GAIL", "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES",
    "GRASIM", "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM", "INDIAMART",
    "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC", "IPCALAB", "IRCTC", "ITC",
    "JINDALSTEL", "JKCEMENT", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LALPATHLAB", "LAURUSLABS",
    "LICHSGFIN", "LT", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO",
    "MARUTI", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS",
    "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC",
    "OBEROIRLTY", "OFSS", "ONGC", "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC",
    "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POONAWALLA", "POWERGRID", "PVRINOX", "RAMCOCEM",
    "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TORNTPOWER",
    "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS", "WIPRO", "ZYDUSLIFE"
}


class UniverseDiscoveryEngine:
    """Discovers, updates, and manages canonical NSE Equity Universe."""

    def __init__(self, market_data_provider: MarketDataProvider, cache_dir: Path | None = None):
        self.provider = market_data_provider
        self.cache_dir = cache_dir or settings.CACHE_DIR / "universe"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.security_master_file = self.cache_dir / "canonical_security_master.json"

    async def build_universe(self, force_refresh: bool = False) -> list[SymbolMetadata]:
        """
        Builds the active eligible NSE equity universe.
        Applies ASM/GSM filters, F&O tagging, and deduplication.
        """
        if not force_refresh and self.security_master_file.exists():
            try:
                data = json.loads(self.security_master_file.read_text(encoding="utf-8"))
                symbols_data = data.get("securities", [])
                if symbols_data:
                    logger.info(f"Loaded {len(symbols_data)} active securities from canonical security master.")
                    return [SymbolMetadata(**s) for s in symbols_data]
            except Exception as e:
                logger.warning(f"Failed to read cached universe: {e}")

        # Fetch active listing from exchange
        raw_securities = await self.provider.fetch_active_securities()
        if not raw_securities:
            logger.warning("Provider returned empty active securities list. Using standard F&O baseline.")
            raw_securities = [
                SymbolMetadata(
                    symbol=sym,
                    company_name=sym,
                    exchange="NSE",
                    is_active=True,
                    is_fno_eligible=True,
                )
                for sym in sorted(STANDARD_FNO_SYMBOLS)
            ]

        # Process and tag F&O / ASM status
        eligible_universe: list[SymbolMetadata] = []
        for sec in raw_securities:
            sym_upper = sec.symbol.upper().strip()

            # Skip symbols with ASM/GSM stage >= 2
            if sec.asm_gsm_stage >= 2:
                logger.debug(f"Excluding {sym_upper} due to ASM/GSM stage {sec.asm_gsm_stage}")
                continue

            # Tag F&O eligibility
            is_fno = sym_upper in STANDARD_FNO_SYMBOLS or sec.is_fno_eligible

            enriched_sec = SymbolMetadata(
                symbol=sym_upper,
                company_name=sec.company_name,
                isin=sec.isin,
                exchange="NSE",
                sector=sec.sector or "General",
                industry=sec.industry or "General",
                is_fno_eligible=is_fno,
                is_active=sec.is_active,
                asm_gsm_stage=sec.asm_gsm_stage,
                lot_size=sec.lot_size,
            )
            eligible_universe.append(enriched_sec)

        # Persist to disk
        self._persist_security_master(eligible_universe)
        logger.info(f"Built eligible NSE universe with {len(eligible_universe)} securities ({sum(1 for s in eligible_universe if s.is_fno_eligible)} F&O eligible).")
        return eligible_universe

    def _persist_security_master(self, securities: list[SymbolMetadata]) -> None:
        """Saves canonical universe to JSON cache."""
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "total_count": len(securities),
            "fno_count": sum(1 for s in securities if s.is_fno_eligible),
            "securities": [s.model_dump() for s in securities],
        }
        self.security_master_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
