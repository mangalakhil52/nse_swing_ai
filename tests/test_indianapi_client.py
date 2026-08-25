from unittest.mock import Mock

from src.data.indianapi_client import IndianAPIClient, IndianAPIConfig


def test_indianapi_client_uses_api_key_and_expected_endpoint():
    client = IndianAPIClient(IndianAPIConfig(api_key="secret", base_url="https://example.test"))
    response = Mock()
    response.json.return_value = {"tickerId": "RELIANCE"}
    response.raise_for_status.return_value = None
    client.session.get = Mock(return_value=response)

    out = client.stock("RELIANCE")

    assert out["tickerId"] == "RELIANCE"
    client.session.get.assert_called_once_with(
        "https://example.test/stock",
        params={"name": "RELIANCE"},
        timeout=15,
    )
    assert client.session.headers["x-api-key"] == "secret"


def test_indianapi_client_supports_historical_stats_and_targets():
    client = IndianAPIClient(IndianAPIConfig(api_key="secret"))
    response = Mock()
    response.json.return_value = {"ok": True}
    response.raise_for_status.return_value = None
    client.session.get = Mock(return_value=response)

    assert client.historical_stats("TCS", "ratios") == {"ok": True}
    assert client.target_price("TCS") == {"ok": True}

    calls = client.session.get.call_args_list
    assert calls[0].kwargs["params"] == {"stock_name": "TCS", "stats": "ratios"}
    assert calls[1].kwargs["params"] == {"stock_id": "TCS"}
