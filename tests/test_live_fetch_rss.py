from src.apps.live_sources import fetch_rss
from src.core.live_schema import LiveSourceConfig


class _Resp:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_fetch_rss_parses_items(monkeypatch) -> None:
    xml = """
<rss><channel>
  <item><title>T1</title><link>https://x/1</link><description>D1</description><pubDate>2026-01-01</pubDate></item>
  <item><title>T2</title><link>https://x/2</link><description>D2</description><pubDate>2026-01-02</pubDate></item>
</channel></rss>
""".strip()

    def fake_get(*_args, **_kwargs):
        return _Resp(xml)

    monkeypatch.setattr("src.apps.live_sources.requests.get", fake_get)
    source = LiveSourceConfig(name="feed", enabled=True, type="rss", url="https://example.com/feed")
    items, raw = fetch_rss(source, query="", max_items=10)
    assert len(items) == 2
    assert items[0].title == "T1"
    assert items[0].url == "https://x/1"
    assert raw.startswith("<rss")
