"""Small stdlib-only API/event contract for the command center.

The production web server can expose these helpers through Flask/FastAPI or
another ASGI/WSGI layer without coupling the UI to the scanner implementation.
"""
from __future__ import annotations

import json
import time
from collections import deque
from threading import Lock
from typing import Any, Deque, Dict, Iterator


class DashboardBus:
    """Thread-safe in-process state/event bus for scanner telemetry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: Deque[dict[str, Any]] = deque(maxlen=500)
        self._state: dict[str, Any] = {
            "connected": True,
            "scan": {
                "universe": 0,
                "filtered": 0,
                "candidates": 0,
                "intel": 0,
                "final": 0,
                "processed": 0,
                "status": "IDLE",
            },
            "agents": {},
            "alerts": [],
        }

    def publish(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        with self._lock:
            self._events.append(event)
            typ = event.get("type")
            if typ == "scan_progress":
                self._state["scan"].update({k: v for k, v in event.items() if k != "type"})
            elif typ == "agent" and event.get("agent"):
                self._state["agents"][event["agent"]] = dict(event)
            elif typ == "connection":
                self._state["connected"] = bool(event.get("connected"))
            elif typ == "alert":
                self._state["alerts"] = [event, *self._state["alerts"]][:25]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def events(self, after_ts: float = 0.0) -> Iterator[dict[str, Any]]:
        with self._lock:
            events = [e for e in self._events if float(e.get("ts", 0)) > after_ts]
        yield from events


BUS = DashboardBus()


def dashboard_payload() -> dict[str, Any]:
    """Return the JSON payload expected by GET /api/dashboard."""
    return BUS.snapshot()


def sse_payload(event: dict[str, Any]) -> str:
    """Serialize one event for GET /api/events (text/event-stream)."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
