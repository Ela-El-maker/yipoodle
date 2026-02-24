import json

from src.apps.live_sources import fetch_rest
from src.core.live_schema import LiveSourceConfig


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_fetch_rest_template_transform(monkeypatch) -> None:
    source = LiveSourceConfig(
        name="yahoo_finance",
        enabled=True,
        type="rest",
        endpoint="https://example.com/chart/{symbol}",
        query_params={"interval": "1m"},
        transform={
            "mode": "template",
            "item_path": "data.values",
            "snippet_template": "Price tick: {value}",
        },
    )

    def fake_get(*_args, **_kwargs):
        return _Resp({"data": {"values": [101.2, 101.4]}})

    monkeypatch.setattr("src.apps.live_sources.requests.get", fake_get)
    items, raw = fetch_rest(source, query="nvda", params={"symbol": "NVDA"}, max_items=5)
    assert len(items) == 2
    assert items[0].source == "yahoo_finance"
    assert "Price tick" in items[0].text
    assert raw
