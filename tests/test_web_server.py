import json
import threading
import urllib.request

from web.api_contract import BUS
from web.server import Handler, ThreadingHTTPServer


def test_dashboard_endpoint_exposes_bus_state():
    BUS.publish({"type": "scan_progress", "universe": 10, "processed": 3, "status": "SCANNING"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/dashboard", timeout=2) as response:
            payload = json.load(response)
        assert payload["scan"]["universe"] == 10
        assert payload["scan"]["processed"] == 3
    finally:
        server.shutdown()
        server.server_close()
