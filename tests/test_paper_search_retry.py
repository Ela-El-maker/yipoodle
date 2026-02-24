import requests

from src.apps import paper_search


class _Resp:
    def __init__(self, status_code: int, text: str = "{}"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}", response=self)


def test_request_get_with_retry_recovers(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Resp(503)
        return _Resp(200, "ok")

    monkeypatch.setattr(paper_search.requests, "get", fake_get)
    monkeypatch.setattr(paper_search.time, "sleep", lambda _s: None)

    resp = paper_search._request_get_with_retry("https://example.org", retries=3)
    assert resp.status_code == 200
    assert calls["n"] == 3
