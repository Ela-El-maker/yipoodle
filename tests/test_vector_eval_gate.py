from src.apps.benchmark import _vector_eval_gate


def test_vector_eval_gate_passes_when_recall_and_speed_targets_met() -> None:
    out = _vector_eval_gate(recall_at_k=0.99, ann_p95_ms=70.0, exact_p95_ms=110.0)
    assert out["gate_enabled"] is True
    assert out["recall_ok"] is True
    assert out["speed_ok"] is True
    assert out["pass"] is True


def test_vector_eval_gate_fails_when_targets_not_met() -> None:
    out = _vector_eval_gate(recall_at_k=0.9, ann_p95_ms=100.0, exact_p95_ms=110.0)
    assert out["pass"] is False
