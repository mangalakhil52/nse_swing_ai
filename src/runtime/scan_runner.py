"""#14T executable runtime bridge for the configured end-to-end scanner."""
from __future__ import annotations
from datetime import date
from src.runtime.end_to_end_scan import build_scan_orchestrator
from src.runtime.production_scan import build_runtime_config, persist_scan
from src.runtime.intelligence_pipeline import build_candidate_discovery, build_decision_pipeline


def run_scan(as_of_date: date, config: dict):
    candidate_discovery = build_candidate_discovery(config)
    decision_pipeline = build_decision_pipeline(config)
    orchestrator = build_scan_orchestrator(config, candidate_discovery, decision_pipeline)
    result = orchestrator.scan(as_of_date)
    persist_scan(result, config.get("output_dir", "data/runs"))
    return result
