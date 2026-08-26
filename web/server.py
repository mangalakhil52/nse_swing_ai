"""Minimal stdlib web adapter for the NSE Swing AI command center.

Serves ./web as static assets and exposes:
  GET /api/dashboard -> current dashboard state
  GET /api/events -> Server-Sent Events stream

The scanner publishes real telemetry through web.api_contract.BUS; the UI never
invents market data.
"""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    from .api_contract import BUS, dashboard_payload, sse_payload
except ImportError:
    from api_contract import BUS, dashboard_payload, sse_payload

ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    server_version = "NSE-Swing-AI/1.0"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", os.getenv("NSE_AI_CORS_ORIGIN", "*"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/dashboard":
            self._send(200, "application/json; charset=utf-8", json.dumps(dashboard_payload()).encode())
            return
        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", os.getenv("NSE_AI_CORS_ORIGIN", "*"))
            self.end_headers()
            last = 0.0
            while True:
                events = list(BUS.events(last))
                for event in events:
                    self.wfile.write(sse_payload(event).encode())
                    self.wfile.flush()
                    last = max(last, float(event.get("ts", last)))
                if not events:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                time.sleep(2)
            return
        if path == "/" or path == "/index.html":
            body = (ROOT / "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
            return
        candidate = (ROOT / path.lstrip("/")).resolve()
        if ROOT in candidate.parents and candidate.is_file():
            content_type = "text/css; charset=utf-8" if candidate.suffix == ".css" else "application/javascript; charset=utf-8"
            self._send(200, content_type, candidate.read_bytes())
            return
        self._send(404, "application/json; charset=utf-8", b'{"error":"not_found"}')


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    serve(host=os.getenv("NSE_AI_HOST", "127.0.0.1"), port=int(os.getenv("NSE_AI_PORT", "8080")))
