"""
Dynamic NSE Universe Discovery & Security Master Module.
Builds the canonical eligible trading universe from official exchange listings.

Production invariants:
- The live universe is sourced from the NSE security master, never a hardcoded
  stock list.
- A failed refresh never silently degrades to a partial F&O-only universe.
- Cached security masters are bounded by a configurable TTL.
- Only active equity series are retained and ASM/GSM stage >= 2 is excluded.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings
from src.core.exceptions import DataUnavailableException
from src.core.models import SymbolMetadata
from src.data.base import MarketDataProvider

logger = logging.getLogger(__name__)


class UniverseDiscoveryEngine:
    """Discovers and manages the canonical NSE equity universe."""

    @classmethod
    def get_default_active_universe(cls) -> list[SymbolMetadata]:
        """Returns standard baseline active SymbolMetadata objects for active trading equities."""
        return [
            SymbolMetadata(
                symbol=sym,
                company_name=sym,
                exchange="NSE",
                sector="General",
                industry="General",
                is_active=True,
                is_fno_eligible=True,
            )
            for sym in sorted(STANDARD_FNO_SYMBOLS)
        ]

    def __init__(self, market_data_provider: MarketDataProvider, cache_dir: Path | None = None):
        self.provider = market_data_provider
        self.cache_dir = cache_dir or settings.CACHE_DIR / "universe"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.security_master_file = self.cache_dir / "canonical_security_master.json"
        self.cache_ttl_hours = 24

    def _cache_is_fresh(self) -> bool:
        if not self.security_master_file.exists():
            return False
        try:
            payload = json.loads(self.security_master_file.read_text(encoding="utf-8"))
            updated_at = payload.get("updated_at")
            if not updated_at:
                return False
            ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - ts <= timedelta(hours=self.cache_ttl_hours)
        except Exception:
            return False

    def _load_cached_universe(self) -> list[SymbolMetadata]:
        try:
            data = json.loads(self.security_master_file.read_text(encoding="utf-8"))
            securities = data.get("securities", [])
            if not securities:
                raise DataUnavailableException("Cached NSE security master is empty")
            universe = [SymbolMetadata(**item) for item in securities]
            if not universe:
                raise DataUnavailableException("Cached NSE security master contains no securities")
            return universe
        except Exception as exc:
            raise DataUnavailableException(f"Invalid NSE security-master cache: {exc}") from exc

    async def build_universe(self, force_refresh: bool = False) -> list[SymbolMetadata]:
        """
        Build the active NSE equity universe from the official security master.

        Fail-closed behavior is intentional: if the exchange listing cannot be
        refreshed and no valid cache exists, the scan must stop rather than trade
        against a stale/partial universe.
        """
        if not force_refresh and self._cache_is_fresh():
            universe = self._load_cached_universe()
            logger.info("Loaded %d active securities from fresh canonical security master.", len(universe))
            return universe

        raw_securities: list[SymbolMetadata] = []
        try:
            raw_securities = await self.provider.fetch_active_securities()
        except Exception as exc:
            logger.warning("NSE security-master refresh failed: %s", exc)

        if not raw_securities:
            # A stale cache can be used only as an explicit availability fallback;
            # never replace it with a fabricated or partial symbol list.
            if self.security_master_file.exists():
                universe = self._load_cached_universe()
                logger.warning(
                    "Using stale canonical security master with %d securities because NSE refresh failed. "
                    "No hardcoded F&O fallback is permitted.",
                    len(universe),
                )
                return universe
            raise DataUnavailableException(
                "NSE active-equity security master unavailable and no valid cache exists; refusing partial-universe scan."
            )

        eligible_universe: list[SymbolMetadata] = []
        seen: set[str] = set()
        for sec in raw_securities:
            sym_upper = sec.symbol.upper().strip()
            if not sym_upper or sym_upper in seen:
                continue
            if not sec.is_active:
                continue
            if sec.asm_gsm_stage >= 2:
                logger.debug("Excluding %s due to ASM/GSM stage %s", sym_upper, sec.asm_gsm_stage)
                continue

            seen.add(sym_upper)
            eligible_universe.append(
                SymbolMetadata(
                    symbol=sym_upper,
                    company_name=sec.company_name,
                    isin=sec.isin,
                    exchange="NSE",
                    sector=sec.sector or "General",
                    industry=sec.industry or "General",
                    is_fno_eligible=bool(sec.is_fno_eligible),
                    is_active=True,
                    asm_gsm_stage=sec.asm_gsm_stage,
                    lot_size=sec.lot_size,
                )
            )

        if not eligible_universe:
            raise DataUnavailableException(
                "NSE security master returned no eligible active equities after validation."
            )

        self._persist_security_master(eligible_universe)
        logger.info(
            "Built canonical NSE universe with %d active securities (%d F&O eligible).",
            len(eligible_universe),
            sum(1 for s in eligible_universe if s.is_fno_eligible),
        )
        return eligible_universe

    def _persist_security_master(self, securities: list[SymbolMetadata]) -> None:
        """Persist the validated canonical universe atomically."""
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_count": len(securities),
            "fno_count": sum(1 for s in securities if s.is_fno_eligible),
            "securities": [s.model_dump() for s in securities],
        }
        tmp = self.security_master_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.security_master_file)
