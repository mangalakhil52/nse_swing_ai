"""
Model Registry & Champion/Challenger Promotion Engine — src/database/model_registry.py

Manages model versioning, shadow performance tracking, and automated promotion of Challenger models to Champion.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    model_id: str
    version: str
    role: str  # CHAMPION, CHALLENGER, ARCHIVED
    weights: dict[str, float]
    created_at: str
    sharpe_ratio: float
    win_rate_pct: float
    max_drawdown_pct: float
    total_shadow_trades: int


class ModelRegistry:
    """Manages active Champion model and evaluates Challenger promotion criteria."""

    def __init__(self, registry_file: Path | None = None):
        self.registry_file = registry_file or settings.DATA_DIR / "model_registry.json"
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        if not self.registry_file.exists():
            default_data = {
                "active_champion": "v1.2.0-champion",
                "models": [
                    {
                        "model_id": "MOD-101",
                        "version": "v1.2.0-champion",
                        "role": "CHAMPION",
                        "weights": {
                            "technical_weight": 0.25,
                            "rs_weight": 0.25,
                            "fundamental_weight": 0.20,
                            "institutional_weight": 0.15,
                            "news_weight": 0.15,
                        },
                        "created_at": datetime.utcnow().isoformat(),
                        "sharpe_ratio": 2.15,
                        "win_rate_pct": 65.0,
                        "max_drawdown_pct": 7.5,
                        "total_shadow_trades": 120,
                    },
                    {
                        "model_id": "MOD-102",
                        "version": "v1.3.0-challenger",
                        "role": "CHALLENGER",
                        "weights": {
                            "technical_weight": 0.28,
                            "rs_weight": 0.26,
                            "fundamental_weight": 0.22,
                            "institutional_weight": 0.14,
                            "news_weight": 0.10,
                        },
                        "created_at": datetime.utcnow().isoformat(),
                        "sharpe_ratio": 2.45,
                        "win_rate_pct": 68.5,
                        "max_drawdown_pct": 6.2,
                        "total_shadow_trades": 45,
                    },
                ],
            }
            self.registry_file.write_text(json.dumps(default_data, indent=2), encoding="utf-8")

    def get_champion_model(self) -> ModelVersion:
        """Returns the current active Champion model."""
        data = json.loads(self.registry_file.read_text(encoding="utf-8"))
        champ_ver = data.get("active_champion")
        for m in data.get("models", []):
            if m.get("version") == champ_ver:
                return ModelVersion(**m)

        # Fallback default champion
        return ModelVersion(
            model_id="MOD-101",
            version="v1.2.0-champion",
            role="CHAMPION",
            weights={"technical_weight": 0.25, "rs_weight": 0.25, "fundamental_weight": 0.20, "institutional_weight": 0.15, "news_weight": 0.15},
            created_at=datetime.utcnow().isoformat(),
            sharpe_ratio=2.15,
            win_rate_pct=65.0,
            max_drawdown_pct=7.5,
            total_shadow_trades=100,
        )

    def evaluate_challenger_promotion(self) -> bool:
        """
        Evaluates if Challenger outperformance justifies promotion to Champion:
          - Sharpe ratio improvement >= +0.25
          - Total shadow trades >= 30
          - Lower max drawdown
        """
        data = json.loads(self.registry_file.read_text(encoding="utf-8"))
        models = [ModelVersion(**m) for m in data.get("models", [])]

        champ = next((m for m in models if m.role == "CHAMPION"), None)
        chal = next((m for m in models if m.role == "CHALLENGER"), None)

        if not champ or not chal:
            return False

        sharpe_diff = chal.sharpe_ratio - champ.sharpe_ratio
        if (
            sharpe_diff >= 0.25
            and chal.total_shadow_trades >= 30
            and chal.max_drawdown_pct <= champ.max_drawdown_pct
        ):
            logger.info(f"🏆 CHALLENGER PROMOTION DETECTED: Promoting {chal.version} to CHAMPION! (Sharpe Improvement: +{sharpe_diff:.2f})")

            # Update roles
            for m in data["models"]:
                if m["version"] == champ.version:
                    m["role"] = "ARCHIVED"
                elif m["version"] == chal.version:
                    m["role"] = "CHAMPION"

            data["active_champion"] = chal.version
            self.registry_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True

        return False
