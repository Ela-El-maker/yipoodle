
from src.apps import vector_service


def test_vector_service_health_missing_metadata(tmp_path) -> None:
    index_path = tmp_path / "idx.json"
    index_path.write_text("{}", encoding="utf-8")

    out = vector_service.vector_service_health(str(index_path))
    assert out["ok"] is False
    assert out["metadata_exists"] is False


def test_vector_service_query_calls_vector_layer(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    idx_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(vector_service, "load_vector_index", lambda *_a, **_k: object())
    monkeypatch.setattr(
        vector_service,
        "query_vector_index",
        lambda *_a, **_k: [("P1:S1", 0.9), ("P2:S1", 0.8)],
    )

    out = vector_service.vector_service_query(
        index_path=str(idx_path),
        question="q",
        top_k=2,
        vector_nprobe=12,
        vector_ef_search=40,
    )
    assert out["ok"] is True
    assert len(out["results"]) == 2


def test_vector_service_build_returns_stats(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    idx_path.write_text("{}", encoding="utf-8")

    class _Idx:
        snippets = []

    class _Bundle:
        index_type = "ivf_flat"
        snippet_ids = ["a", "b"]
        dimension = 384
        shard_count = 2
        ann_params = {"nlist": 128}
        build_stats = {"train_size": 2}

    monkeypatch.setattr(vector_service, "load_index", lambda *_a, **_k: _Idx())
    monkeypatch.setattr(vector_service, "build_vector_index", lambda *_a, **_k: _Bundle())
    monkeypatch.setattr(vector_service, "save_vector_index", lambda *_a, **_k: None)

    out = vector_service.vector_service_build(index_path=str(idx_path), vector_index_type="ivf_flat", vector_shards=2)
    assert out["ok"] is True
    assert out["vector_index_type"] == "ivf_flat"
    assert out["vector_rows"] == 2
