import json
import threading
import urllib.request

from src.apps import vector_service


class _Loaded:
    index_type = "ivf_flat"
    shard_count = 2
    snippet_ids = ["P1:S1", "P2:S1"]


def test_vector_service_http_health_and_query(monkeypatch, tmp_path) -> None:
    idx_path = tmp_path / "idx.json"
    idx_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(vector_service, "load_vector_index", lambda *_a, **_k: _Loaded())

    seen = {}

    def _fake_query(_loaded, question, top_k, model_name_override=None, nprobe=16, ef_search=64):
        seen["question"] = question
        seen["top_k"] = top_k
        seen["model_name_override"] = model_name_override
        seen["nprobe"] = nprobe
        seen["ef_search"] = ef_search
        return [("P1:S1", 0.9), ("P2:S1", 0.8)]

    monkeypatch.setattr(vector_service, "query_vector_index", _fake_query)

    server, info = vector_service.start_vector_service_server(
        index_path=str(idx_path),
        host="127.0.0.1",
        port=0,
        embedding_model="fake/model",
        vector_nprobe=12,
        vector_ef_search=40,
    )

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://{info['host']}:{info['port']}"

        with urllib.request.urlopen(base + "/health", timeout=2.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["vector_index_type"] == "ivf_flat"

        req = urllib.request.Request(
            base + "/query",
            data=json.dumps({"question": "q1", "top_k": 2}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            qbody = json.loads(resp.read().decode("utf-8"))

        assert qbody["ok"] is True
        assert len(qbody["results"]) == 2
        assert seen["question"] == "q1"
        assert seen["top_k"] == 2
        assert seen["model_name_override"] == "fake/model"
        assert seen["nprobe"] == 12
        assert seen["ef_search"] == 40
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=2.0)
