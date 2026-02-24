from src.apps.live_snapshot_store import load_cached_snapshot, save_snapshot, snapshot_to_snippets
from src.core.live_schema import LiveItem


def test_snapshot_roundtrip_and_snippets(tmp_path) -> None:
    items = [
        LiveItem(id="a", url="https://x", text="hello world", source="demo"),
        LiveItem(id="b", url="https://y", text="<b>content</b>", source="demo"),
    ]
    snap, path = save_snapshot(
        root_dir=str(tmp_path),
        source="demo",
        query="q",
        params={},
        items=items,
        persist_raw=True,
        raw_payload="raw",
    )
    assert path.endswith(".snapshot.json")

    cached = load_cached_snapshot(root_dir=str(tmp_path), source="demo", query="q", params={}, ttl_sec=60)
    assert cached is not None
    assert cached.snapshot_id == snap.snapshot_id

    snippets = snapshot_to_snippets(cached)
    assert snippets
    assert snippets[0].snippet_id.startswith("SNAP:")
    assert "content" in snippets[1].text
