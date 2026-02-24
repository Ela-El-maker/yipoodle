from src.apps.kb_store import connect_kb, ensure_topic, init_kb, list_topics, query_claims, upsert_claim


def test_kb_store_init_and_upsert(tmp_path) -> None:
    db = tmp_path / "kb.db"
    init_kb(str(db))
    with connect_kb(str(db)) as conn:
        tid = ensure_topic(conn, "finance_markets")
        claim_id, action, _, _ = upsert_claim(
            conn,
            topic_id=tid,
            claim_text="Transformers can overfit short horizons",
            canonical_hash="h1",
            confidence=0.6,
            run_id="r1",
            now_iso="2026-01-01T00:00:00+00:00",
        )
        assert action == "added"
        claim_id2, action2, _, _ = upsert_claim(
            conn,
            topic_id=tid,
            claim_text="Transformers can overfit short horizons",
            canonical_hash="h1",
            confidence=0.7,
            run_id="r2",
            now_iso="2026-01-02T00:00:00+00:00",
        )
        conn.commit()
        assert claim_id == claim_id2
        assert action2 == "updated"
        assert "finance_markets" in list_topics(conn)
        rows = query_claims(conn, query="transformers", topic="finance_markets", top_k=5)
        assert rows
        assert rows[0]["claim_id"] == claim_id
