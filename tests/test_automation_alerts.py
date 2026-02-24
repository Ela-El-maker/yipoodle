from __future__ import annotations

from src.apps.automation import collect_alert_events, dispatch_alerts


def _base_summary() -> dict:
    return {
        "run_id": "20260222T220000Z",
        "created_utc": "2026-02-22T22:00:00Z",
        "any_topic_failure": False,
        "sync": {"source_errors": 0},
        "corpus_health": {"healthy": True, "reasons": [], "warnings": []},
        "topics": [{"name": "t1", "validate_ok": True, "evidence_usage": 0.8, "report_path": "r.md"}],
    }


def _base_config() -> dict:
    return {
        "thresholds": {"warn_evidence_usage_below": 0.7},
        "alerts": {
            "enabled": True,
            "webhook_url": "https://example.invalid/hook",
            "webhook_timeout_sec": 5,
            "webhook_headers": {"X-Test": "1"},
            "on_corpus_unhealthy": True,
            "on_topic_validation_failed": True,
            "on_source_errors": True,
        },
    }


def test_collect_alert_events_detects_failures() -> None:
    summary = _base_summary()
    summary["corpus_health"] = {"healthy": False, "reasons": ["x"], "warnings": []}
    summary["topics"] = [{"name": "t1", "validate_ok": False, "evidence_usage": 0.2, "report_path": "r.md"}]
    summary["sync"] = {"source_errors": 1}
    summary["any_topic_failure"] = True
    events = collect_alert_events(summary, _base_config())
    codes = {e["code"] for e in events}
    assert "corpus_unhealthy" in codes
    assert "topic_validation_failed" in codes
    assert "low_evidence_usage" in codes
    assert "source_errors" in codes
    assert "topic_runtime_failure" in codes


def test_dispatch_alerts_disabled_returns_without_send() -> None:
    cfg = _base_config()
    cfg["alerts"]["enabled"] = False
    out = dispatch_alerts(_base_summary(), cfg)
    assert out["enabled"] is False
    assert out["sent"] is False


def test_dispatch_alerts_sends_webhook(monkeypatch) -> None:
    summary = _base_summary()
    summary["topics"][0]["evidence_usage"] = 0.1
    sent = {}

    class _Resp:
        status_code = 200

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        sent["url"] = url
        sent["payload"] = json
        sent["headers"] = headers
        sent["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("src.apps.automation.requests.post", _fake_post)
    out = dispatch_alerts(summary, _base_config())
    assert out["sent"] is True
    assert out["transport"] == "webhook"
    assert sent["url"] == "https://example.invalid/hook"
    assert sent["payload"]["run_id"] == summary["run_id"]


def test_dispatch_alerts_webhook_error(monkeypatch) -> None:
    summary = _base_summary()
    summary["topics"][0]["evidence_usage"] = 0.1

    class _Resp:
        status_code = 500

    monkeypatch.setattr("src.apps.automation.requests.post", lambda *args, **kwargs: _Resp())
    out = dispatch_alerts(summary, _base_config())
    assert out["sent"] is False
    assert out["error"] == "webhook_http_500"


def test_dispatch_alerts_sends_email_via_smtp_ssl(monkeypatch) -> None:
    summary = _base_summary()
    summary["topics"][0]["evidence_usage"] = 0.1
    cfg = _base_config()
    cfg["alerts"]["webhook_url"] = None
    cfg["alerts"]["email_enabled"] = True
    cfg["alerts"]["email_to"] = ["team@example.com"]
    cfg["alerts"]["smtp_username"] = "alerts@gmail.com"
    cfg["alerts"]["smtp_password_env"] = "GMAIL_APP_PASSWORD"
    sent = {"login": None, "to": None}

    class _SMTP:
        def __init__(self, host, port, timeout=None) -> None:
            assert host == "smtp.gmail.com"
            assert port == 465
            assert timeout == 5

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def login(self, username, password) -> None:
            sent["login"] = (username, password)

        def send_message(self, msg) -> None:  # noqa: ANN001
            sent["to"] = msg["To"]

    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-pass")
    monkeypatch.setattr("src.apps.automation.smtplib.SMTP_SSL", _SMTP)

    out = dispatch_alerts(summary, cfg)
    assert out["sent"] is True
    assert "email" in out["transport"]
    assert sent["login"] == ("alerts@gmail.com", "app-pass")
    assert sent["to"] == "team@example.com"


def test_collect_alert_events_includes_high_monitoring_and_skips_medium_digest() -> None:
    summary = _base_summary()
    summary["monitoring"] = {
        "events": [
            {
                "severity": "critical",
                "code": "monitor_trigger_high",
                "message": "x",
                "details": {},
            },
            {
                "severity": "warning",
                "code": "monitor_trigger_medium_digest_queued",
                "message": "y",
                "details": {},
            },
        ]
    }
    events = collect_alert_events(summary, _base_config())
    codes = {e["code"] for e in events}
    assert "monitor_trigger_high" in codes
    assert "monitor_trigger_medium_digest_queued" not in codes
