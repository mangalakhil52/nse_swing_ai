"""Minimal model/version registry for reproducible research and live signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json


@dataclass(frozen=True)
class ModelVersion:
    model_name: str
    version: str
    feature_schema_hash: str
    training_end: str
    created_at: str


def feature_schema_hash(feature_names: list[str]) -> str:
    payload = json.dumps(sorted(set(feature_names)), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def create_version(model_name: str, version: str, feature_names: list[str], training_end: str) -> ModelVersion:
    return ModelVersion(
        model_name=model_name,
        version=version,
        feature_schema_hash=feature_schema_hash(feature_names),
        training_end=training_end,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
