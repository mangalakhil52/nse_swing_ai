"""Optional scanner telemetry hooks for the web command center.

Telemetry is deliberately best-effort: a dashboard failure must never stop a
market-data scan. Importing this module does not require the web server.
"""
from __future__ import annotations

from typing import Any


def publish(event: dict[str, Any]) -> None:
    try:
        from web.api_contract import BUS
        BUS.publish(event)
    except Exception:
        # Observability must never become a trading-path dependency.
        return


def scan_progress(**fields: Any) -> None:
    publish({"type": "scan_progress", **fields})


def agent(agent_name: str, **fields: Any) -> None:
    publish({"type": "agent", "agent": agent_name, **fields})


def alert(message: str, severity: str = "amber") -> None:
    publish({"type": "alert", "message": message, "severity": severity})
