from __future__ import annotations

import pytest

from src.core.exceptions import DataUnavailableException
from src.data.nse_index_provider import NseIndexDataProvider


class FakeResponse:
    def __init__(self, status_code: int, content_type: str, text: str, payload=None):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is not None:
            return self._payload
        raise ValueError("invalid json")


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def get(self, *args, **kwargs):
        self.calls += 1
        return next(self.responses)


@pytest.mark.asyncio
async def test_http_200_html_is_retried_then_valid_json_is_accepted():
    provider = NseIndexDataProvider()
    client = FakeClient([
        FakeResponse(200, "text/html", "<html>blocked</html>"),
        FakeResponse(
            200,
            "application/json",
            '{"data":{"indexCloseOnlineRecords":[]}}',
            {"data": {"indexCloseOnlineRecords": [{"EOD_TIMESTAMP": "04-Sep-2026", "EOD_OPEN_INDEX_VAL": "100", "EOD_HIGH_INDEX_VAL": "110", "EOD_LOW_INDEX_VAL": "90", "EOD_CLOSE_INDEX_VAL": "105"}]}},
        ),
    ])
    provider._session = client
    provider.max_attempts = 2

    payload = await provider._get_json("https://nse.example/api", "https://nse.example/")

    assert payload["data"]["indexCloseOnlineRecords"]
    assert client.calls == 2


@pytest.mark.asyncio
async def test_repeated_http_200_html_fails_closed():
    provider = NseIndexDataProvider()
    provider._session = FakeClient([
        FakeResponse(200, "text/html", "<html>blocked</html>"),
        FakeResponse(200, "text/html", "<html>blocked</html>"),
    ])
    provider.max_attempts = 2

    with pytest.raises(DataUnavailableException, match="non-JSON"):
        await provider._get_json("https://nse.example/api", "https://nse.example/")


@pytest.mark.asyncio
async def test_invalid_json_with_json_content_type_fails_closed():
    provider = NseIndexDataProvider()
    provider._session = FakeClient([
        FakeResponse(200, "application/json", "not-json"),
    ])
    provider.max_attempts = 1

    with pytest.raises(DataUnavailableException, match="invalid JSON"):
        await provider._get_json("https://nse.example/api", "https://nse.example/")
