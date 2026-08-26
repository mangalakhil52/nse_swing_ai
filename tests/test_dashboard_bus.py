from web.api_contract import DashboardBus, sse_payload


def test_dashboard_bus_tracks_scan_and_agents():
    bus = DashboardBus()
    bus.publish({"type": "scan_progress", "universe": 2557, "processed": 120, "status": "SCANNING"})
    bus.publish({"type": "agent", "agent": "TECHNICAL", "progress": 50, "processed": 120, "decision": "PASS"})
    state = bus.snapshot()
    assert state["scan"]["universe"] == 2557
    assert state["scan"]["processed"] == 120
    assert state["agents"]["TECHNICAL"]["decision"] == "PASS"


def test_sse_payload_is_event_stream():
    payload = sse_payload({"type": "alert", "message": "test"})
    assert payload.startswith("data: ")
    assert payload.endswith("\n\n")
