from src.apps.monitoring_rules import evaluate_monitor_rules


def test_monitoring_rules_all_trigger_types() -> None:
    delta = {
        "new_sources_count": 2,
        "kb_diff_counts": {"added": 3, "updated": 0, "disputed": 1},
        "metrics_current": {"evidence_usage": 0.5},
        "report_text": {"synthesis": "There is outage risk in this topic."},
    }
    triggers = [
        {"id": "t1", "type": "new_documents", "severity": "medium", "min_new_sources": 1},
        {"id": "t2", "type": "kb_diff_count", "severity": "high", "min_added_claims": 2},
        {"id": "t3", "type": "metric_threshold", "severity": "high", "metric": "evidence_usage", "op": "lt", "value": 0.6},
        {"id": "t4", "type": "keyword_presence", "severity": "high", "target": "synthesis", "keywords": ["outage"]},
    ]

    out, errs = evaluate_monitor_rules(topic="finance", run_id="r1", delta=delta, triggers=triggers)
    assert not errs
    assert len(out) == 4
    assert all(x["fired"] is True for x in out)


def test_monitoring_rules_unknown_type_returns_error() -> None:
    out, errs = evaluate_monitor_rules(
        topic="t", run_id="r", delta={}, triggers=[{"id": "x", "type": "nope"}]
    )
    assert out == []
    assert errs and "unknown trigger type" in errs[0]
