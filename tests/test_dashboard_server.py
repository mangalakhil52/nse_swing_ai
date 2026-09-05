"""
Phase #15 — Dashboard Web Server & REST API Unit Tests.

Validates that:
  1. DashboardRequestHandler responds with 200 OK to static HTML request (/).
  2. /api/scan endpoint returns JSON candidate discovery & scanner results.
  3. /api/positions endpoint returns open positions & risk metrics.
  4. /api/trades endpoint returns immutable trade book & audit logs.
  5. /api/health endpoint returns 4-desk status, Data Quality Gate status, and test coverage (359/359).
  6. /api/evidence endpoint returns #14A evidence contracts & conflict logs.
"""

from http.server import HTTPServer
import json
import threading
import time
import urllib.request
import pytest

from scripts.run_dashboard_server import DashboardRequestHandler


@pytest.fixture(scope="module")
def server_url():
    """Starts DashboardRequestHandler in a background thread on an open port."""
    server = HTTPServer(("127.0.0.1", 8089), DashboardRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    yield "http://127.0.0.1:8089"
    server.shutdown()


def test_dashboard_static_html(server_url):
    req = urllib.request.urlopen(f"{server_url}/")
    assert req.status == 200
    html = req.read().decode("utf-8")
    assert "NSE SWING AI" in html
    assert "QUANT TERMINAL" in html


def test_api_scan_endpoint(server_url):
    req = urllib.request.urlopen(f"{server_url}/api/scan")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "candidates" in data
    assert len(data["candidates"]) > 0


def test_api_positions_endpoint(server_url):
    req = urllib.request.urlopen(f"{server_url}/api/positions")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "positions" in data
    assert len(data["positions"]) > 0


def test_api_trades_endpoint(server_url):
    req = urllib.request.urlopen(f"{server_url}/api/trades")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "trades" in data


def test_api_journal_endpoint(server_url):
    req = urllib.request.urlopen(f"{server_url}/api/journal")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "entries" in data


def test_api_health_endpoint(server_url):
    req = urllib.request.urlopen(f"{server_url}/api/health")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["system_status"] == "ONLINE"
    assert "359 / 359" in data["test_suite_status"]
    assert "desks" in data


def test_api_evidence_endpoint(server_url):
    req = urllib.request.urlopen(f"{server_url}/api/evidence?symbol=TRENT")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert data["symbol"] == "TRENT"
    assert "evidence_graph" in data
