from unittest.mock import Mock

from src.data.truedata_client import TrueDataClient, TrueDataConfig


def test_truedata_client_uses_configured_credentials_and_endpoint():
    client = TrueDataClient(TrueDataConfig(username="u", password="p", base_url="https://example.test"))
    response = Mock()
    response.json.return_value = {"ok": True}
    response.raise_for_status.return_value = None
    client.session.get = Mock(return_value=response)

    assert client.snapshot("RELIANCE") == {"ok": True}
    client.session.get.assert_called_once_with(
        "https://example.test/snapshot",
        params={"symbol": "RELIANCE"},
        auth=("u", "p"),
        timeout=15,
    )
    client.close()


def test_truedata_historical_request_is_explicitly_scoped():
    client = TrueDataClient(TrueDataConfig(username="u", password="p"))
    response = Mock()
    response.json.return_value = []
    response.raise_for_status.return_value = None
    client.session.get = Mock(return_value=response)

    client.historical("TCS", "2026-08-01", "2026-08-25", interval="eod")
    kwargs = client.session.get.call_args.kwargs
    assert kwargs["params"] == {
        "symbol": "TCS",
        "startdate": "2026-08-01",
        "enddate": "2026-08-25",
        "interval": "eod",
    }
    client.close()
