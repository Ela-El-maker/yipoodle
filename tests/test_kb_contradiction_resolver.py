from __future__ import annotations

from pathlib import Path
import json

from src.apps.kb_confidence import canonical_hash
from src.apps.kb_contradiction_resolver import run_kb_contradiction_resolver
from src.apps.kb_store import connect_kb, ensure_topic, init_kb, insert_contradiction, upsert_claim


def _seed_disputed_pair(db: str) -> tuple[int, int]:
    init_kb(db)
    with connect_kb(db) as conn:
        conn.execute("BEGIN")
        tid = ensure_topic(conn, "finance")
        now = "2026-02-24T00:00:00+00:00"
        a, _, _, _ = upsert_claim(
            conn,
            topic_id=tid,
            claim_text="Model improves forecasting accuracy under drift",
            canonical_hash=canonical_hash("Model improves forecasting accuracy under drift"),
            confidence=0.8,
            run_id="r1",
            now_iso=now,
        )
        b, _, _, _ = upsert_claim(
            conn,
            topic_id=tid,
            claim_text="Model does not improve forecasting accuracy under drift",
            canonical_hash=canonical_hash("Model does not improve forecasting accuracy under drift"),
            confidence=0.8,
            run_id="r1",
            now_iso=now,
        )
        conn.execute("UPDATE kb_claim SET status='disputed' WHERE id IN (?, ?)", (a, b))
        insert_contradiction(
            conn,
            claim_id_a=a,
            claim_id_b=b,
            score=0.8,
            run_id="r1",
            reason="negation_overlap_conflict",
            now_iso=now,
        )
        conn.commit()
    return a, b


def test_kb_contradiction_resolver_resolves_pair(tmp_path, monkeypatch) -> None:
    db = str(tmp_path / "kb.db")
    claim_a, claim_b = _seed_disputed_pair(db)

    def _fake_research(*, out_path, **kwargs):  # noqa: ANN001
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# resolution\n", encoding="utf-8")
        # Evidence strongly supports claim A text terms.
        evidence = {
            "question": "x",
            "items": [
                {
                    "paper_id": "p1",
                    "snippet_id": "Pp1:S1",
                    "score": 1.2,
                    "section": "results",
                    "text": "forecasting accuracy improves under drift with this model",
                }
            ],
        }
        p.with_suffix(".evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        return str(p)

    monkeypatch.setattr("src.apps.kb_contradiction_resolver.run_research", _fake_research)

    out = run_kb_contradiction_resolver(
        kb_db=db,
        topic="finance",
        index_path=str(tmp_path / "idx.json"),
        out_dir=str(tmp_path / "res"),
        support_margin=0.01,
    )
    assert out["pairs_seen"] == 1
    assert out["pairs_resolved"] == 1
    assert Path(out["summary_path"]).exists()

    with connect_kb(db) as conn:
        rows = conn.execute("SELECT id, status FROM kb_claim WHERE id IN (?, ?)", (claim_a, claim_b)).fetchall()
    status = {int(r["id"]): str(r["status"]) for r in rows}
    assert status[claim_a] == "active"
    assert status[claim_b] == "superseded"


def test_kb_contradiction_resolver_respects_margin(tmp_path, monkeypatch) -> None:
    db = str(tmp_path / "kb.db")
    _seed_disputed_pair(db)

    def _fake_research(*, out_path, **kwargs):  # noqa: ANN001
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# resolution\n", encoding="utf-8")
        evidence = {
            "question": "x",
            "items": [
                {
                    "paper_id": "p1",
                    "snippet_id": "Pp1:S1",
                    "score": 0.1,
                    "section": "abstract",
                    "text": "model forecasting results under drift",
                }
            ],
        }
        p.with_suffix(".evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        return str(p)

    monkeypatch.setattr("src.apps.kb_contradiction_resolver.run_research", _fake_research)
    out = run_kb_contradiction_resolver(
        kb_db=db,
        topic="finance",
        index_path=str(tmp_path / "idx.json"),
        out_dir=str(tmp_path / "res"),
        support_margin=0.8,
    )
    assert out["pairs_resolved"] == 0
    assert out["pairs_unresolved"] == 1
