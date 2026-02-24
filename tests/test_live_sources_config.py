from src.apps.live_sources import live_snapshot_config, live_sources_config


def test_live_sources_config_parses_defaults() -> None:
    cfg = {
        "live_sources": {
            "demo": {
                "enabled": True,
                "type": "rest",
                "endpoint": "https://example.com/api",
            }
        }
    }
    got = live_sources_config(cfg)
    assert "demo" in got
    row = got["demo"]
    assert row.enabled is True
    assert row.type == "rest"
    assert row.timeout_sec == 20
    assert row.cache_ttl_sec == 300
    assert row.rate_limit_rpm == 30


def test_live_snapshot_config_defaults() -> None:
    got = live_snapshot_config({})
    assert got["root_dir"] == "data/live_snapshots"
    assert got["retention_days"] == 30
    assert got["persist_raw"] is True
    assert got["max_body_bytes"] == 2_000_000
