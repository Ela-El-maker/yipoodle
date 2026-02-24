from __future__ import annotations

import json

from src.apps.monitoring_diff import build_monitor_delta, find_previous_successful_topic_state


def test_find_previous_successful_topic_state_and_delta(tmp_path) -> None:
    audit_dir = tmp_path / "runs" / "audit"
    prev_run = audit_dir / "runs" / "20260224T100000Z"
    cur_run = audit_dir / "runs" / "20260224T110000Z"
    prev_run.mkdir(parents=True)
    cur_run.mkdir(parents=True)

    prev_report = tmp_path / "prev.md"
    prev_report.write_text("synth", encoding="utf-8")
    prev_evidence = prev_report.with_suffix(".evidence.json")
    prev_evidence.write_text(json.dumps({"items": [{"snippet_id": "P1:S1"}]}), encoding="utf-8")
    prev_metrics = tmp_path / "prev.metrics.json"
    prev_metrics.write_text(json.dumps({"evidence_usage": 0.8}), encoding="utf-8")

    (prev_run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "20260224T100000Z",
                "topics": [
                    {
                        "name": "finance",
                        "validate_ok": True,
                        "report_path": str(prev_report),
                        "metrics_path": str(prev_metrics),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rid, topic_state = find_previous_successful_topic_state(
        audit_dir=str(audit_dir),
        current_run_id="20260224T110000Z",
        topic_name="finance",
    )
    assert rid == "20260224T100000Z"
    assert topic_state is not None

    cur_report = tmp_path / "cur.md"
    cur_report.write_text("## Synthesis\nnew outage\n## Gaps\nnone", encoding="utf-8")
    cur_evidence = cur_report.with_suffix(".evidence.json")
    cur_evidence.write_text(json.dumps({"items": [{"snippet_id": "P1:S1"}, {"snippet_id": "P2:S1"}]}), encoding="utf-8")
    cur_metrics = tmp_path / "cur.metrics.json"
    cur_metrics.write_text(json.dumps({"evidence_usage": 0.5}), encoding="utf-8")

    delta = build_monitor_delta(
        run_id="20260224T110000Z",
        topic_name="finance",
        topic_state={"report_path": str(cur_report), "metrics_path": str(cur_metrics)},
        baseline_run_id=rid,
        baseline_topic_state=topic_state,
    )
    assert delta["new_sources_count"] == 1
    assert delta["metrics_current"]["evidence_usage"] == 0.5
    assert "outage" in delta["report_text"]["full_report"]
