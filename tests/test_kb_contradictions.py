from src.apps.kb_store import connect_kb, ensure_topic, find_potential_contradictions, init_kb, upsert_claim


def test_kb_contradiction_detection_marks_candidates(tmp_path) -> None:
    db = tmp_path / "kb.db"
    init_kb(str(db))
    with connect_kb(str(db)) as conn:
        tid = ensure_topic(conn, "finance")
        cid1, _, _, _ = upsert_claim(
            conn,
            topic_id=tid,
            claim_text="Model improves forecasting accuracy under drift",
            canonical_hash="h1",
            confidence=0.7,
            run_id="r1",
            now_iso="2026-01-01T00:00:00+00:00",
        )
        cid2, _, _, _ = upsert_claim(
            conn,
            topic_id=tid,
            claim_text="Model does not improve forecasting accuracy under drift",
            canonical_hash="h2",
            confidence=0.6,
            run_id="r2",
            now_iso="2026-01-02T00:00:00+00:00",
        )
        conn.commit()
        out = find_potential_contradictions(conn, topic_id=tid, claim_id=cid2, claim_text="Model does not improve forecasting accuracy under drift")
        assert any(row[0] == cid1 for row in out)
