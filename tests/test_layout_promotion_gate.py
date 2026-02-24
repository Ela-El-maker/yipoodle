import json

from src.apps.layout_promotion import run_layout_promotion_gate


def test_layout_promotion_gate_promotes_when_v2_non_regressive(monkeypatch, tmp_path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"papers": []}), encoding="utf-8")
    state = tmp_path / "layout_state.json"

    def _fake_extract(*args, **kwargs):
        return {"created": 1}

    def _fake_eval(corpus_dir, gold_path):
        if corpus_dir.endswith("legacy"):
            return {
                "summary": {"weighted_score": 0.82},
                "results": [{"checks": [{"type": "ordered_contains", "passed": True}, {"type": "page_nonempty_ratio", "passed": True}]}],
            }
        return {
            "summary": {"weighted_score": 0.84},
            "results": [{"checks": [{"type": "ordered_contains", "passed": True}, {"type": "page_nonempty_ratio", "passed": True}]}],
        }

    monkeypatch.setattr("src.apps.layout_promotion.extract_from_papers_dir", _fake_extract)
    monkeypatch.setattr("src.apps.layout_promotion.evaluate_extraction_against_gold", _fake_eval)

    out = run_layout_promotion_gate(
        papers_dir=str(papers),
        gold_path=str(gold),
        state_path=str(state),
        min_weighted_score=0.75,
    )
    assert out["promoted"] is True
    assert out["recommended_layout_engine"] == "v2"
    assert state.exists()


def test_layout_promotion_gate_stays_shadow_on_regression(monkeypatch, tmp_path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "a.pdf").write_bytes(b"%PDF-1.4")
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"papers": []}), encoding="utf-8")

    monkeypatch.setattr("src.apps.layout_promotion.extract_from_papers_dir", lambda *a, **k: {"created": 1})

    def _fake_eval(corpus_dir, gold_path):
        if corpus_dir.endswith("legacy"):
            return {
                "summary": {"weighted_score": 0.9},
                "results": [{"checks": [{"type": "ordered_contains", "passed": True}, {"type": "page_nonempty_ratio", "passed": True}]}],
            }
        return {
            "summary": {"weighted_score": 0.7},
            "results": [{"checks": [{"type": "ordered_contains", "passed": False}, {"type": "page_nonempty_ratio", "passed": True}]}],
        }

    monkeypatch.setattr("src.apps.layout_promotion.evaluate_extraction_against_gold", _fake_eval)

    out = run_layout_promotion_gate(
        papers_dir=str(papers),
        gold_path=str(gold),
        state_path=str(tmp_path / "state.json"),
        min_weighted_score=0.75,
    )
    assert out["promoted"] is False
    assert out["recommended_layout_engine"] == "shadow"
