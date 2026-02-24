from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import threading

from src.apps.retrieval import derive_vector_paths, load_index
from src.apps.vector_index import (
    DEFAULT_EMBEDDING_MODEL,
    build_vector_index,
    load_vector_index,
    query_vector_index,
    save_vector_index,
)


def vector_service_build(
    index_path: str,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    vector_index_type: str = "flat",
    vector_nlist: int = 1024,
    vector_m: int = 32,
    vector_ef_construction: int = 200,
    vector_shards: int = 1,
    vector_train_sample_size: int = 200000,
) -> dict[str, object]:
    idx = load_index(index_path)
    vec_idx_path, vec_meta_path = derive_vector_paths(index_path, vector_index_path, vector_metadata_path)
    bundle = build_vector_index(
        idx.snippets,
        model_name=embedding_model,
        batch_size=batch_size,
        index_type=vector_index_type,
        nlist=vector_nlist,
        m=vector_m,
        ef_construction=vector_ef_construction,
        shards=vector_shards,
        train_sample_size=vector_train_sample_size,
    )
    save_vector_index(bundle, vec_idx_path, vec_meta_path)
    return {
        "ok": True,
        "index_path": str(Path(index_path).resolve()),
        "vector_index_path": str(Path(vec_idx_path).resolve()),
        "vector_metadata_path": str(Path(vec_meta_path).resolve()),
        "vector_index_type": bundle.index_type,
        "vector_rows": len(bundle.snippet_ids),
        "vector_dim": bundle.dimension,
        "vector_shards": int(bundle.shard_count or 1),
        "vector_ann_params": bundle.ann_params or {},
        "vector_build_stats": bundle.build_stats or {},
    }


def vector_service_query(
    index_path: str,
    question: str,
    top_k: int = 8,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_nprobe: int = 16,
    vector_ef_search: int = 64,
) -> dict[str, object]:
    vec_idx_path, vec_meta_path = derive_vector_paths(index_path, vector_index_path, vector_metadata_path)
    loaded = load_vector_index(vec_idx_path, vec_meta_path)
    ranked = query_vector_index(
        loaded,
        question=question,
        top_k=top_k,
        model_name_override=embedding_model,
        nprobe=vector_nprobe,
        ef_search=vector_ef_search,
    )
    return {
        "ok": True,
        "question": question,
        "top_k": int(top_k),
        "vector_index_type": str(getattr(loaded, "index_type", "flat")),
        "vector_shards": int(getattr(loaded, "shard_count", 1) or 1),
        "results": [{"snippet_id": sid, "score": float(score)} for sid, score in ranked],
    }


def vector_service_health(
    index_path: str,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
) -> dict[str, object]:
    vec_idx_path, vec_meta_path = derive_vector_paths(index_path, vector_index_path, vector_metadata_path)
    meta_exists = Path(vec_meta_path).exists()
    idx_exists = Path(vec_idx_path).exists()
    payload: dict[str, object] = {
        "ok": False,
        "vector_index_path": str(Path(vec_idx_path).resolve()),
        "vector_metadata_path": str(Path(vec_meta_path).resolve()),
        "index_exists": idx_exists,
        "metadata_exists": meta_exists,
    }
    if not meta_exists:
        payload["reason"] = "missing_metadata"
        return payload

    meta = json.loads(Path(vec_meta_path).read_text(encoding="utf-8"))
    payload.update(
        {
            "metadata_version": int(meta.get("version", 1)),
            "vector_index_type": str(meta.get("index_type", "flat")),
            "vector_rows": int(meta.get("snippet_count", 0)),
            "vector_shards": int(meta.get("shard_count", 1) or 1),
        }
    )
    try:
        _ = load_vector_index(vec_idx_path, vec_meta_path)
    except Exception as exc:
        payload["reason"] = f"load_failed:{exc}"
        return payload

    payload["ok"] = True
    payload["reason"] = "ready"
    return payload


def _make_handler(
    *,
    loaded,
    default_embedding_model: str,
    default_nprobe: int,
    default_ef_search: int,
):
    class _Handler(BaseHTTPRequestHandler):
        def _write_json(self, code: int, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "reason": "ready",
                        "vector_index_type": str(getattr(loaded, "index_type", "flat")),
                        "vector_shards": int(getattr(loaded, "shard_count", 1) or 1),
                        "vector_rows": len(getattr(loaded, "snippet_ids", []) or []),
                    },
                )
                return
            self._write_json(404, {"ok": False, "reason": "not_found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/query":
                self._write_json(404, {"ok": False, "reason": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            body = self.rfile.read(max(0, content_length)) if content_length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._write_json(400, {"ok": False, "reason": "invalid_json"})
                return
            question = str(payload.get("question", "")).strip()
            if not question:
                self._write_json(400, {"ok": False, "reason": "missing_question"})
                return
            try:
                top_k = max(1, int(payload.get("top_k", 8)))
            except Exception:
                top_k = 8
            try:
                nprobe = max(1, int(payload.get("vector_nprobe", default_nprobe)))
            except Exception:
                nprobe = default_nprobe
            try:
                ef_search = max(8, int(payload.get("vector_ef_search", default_ef_search)))
            except Exception:
                ef_search = default_ef_search
            embedding_model = str(payload.get("embedding_model") or default_embedding_model)
            ranked = query_vector_index(
                loaded,
                question=question,
                top_k=top_k,
                model_name_override=embedding_model,
                nprobe=nprobe,
                ef_search=ef_search,
            )
            self._write_json(
                200,
                {
                    "ok": True,
                    "question": question,
                    "top_k": top_k,
                    "vector_index_type": str(getattr(loaded, "index_type", "flat")),
                    "vector_shards": int(getattr(loaded, "shard_count", 1) or 1),
                    "results": [{"snippet_id": sid, "score": float(score)} for sid, score in ranked],
                },
            )

        def log_message(self, format, *args):  # noqa: A003
            # Keep CLI output clean in normal usage.
            return

    return _Handler


def start_vector_service_server(
    *,
    index_path: str,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_nprobe: int = 16,
    vector_ef_search: int = 64,
) -> tuple[ThreadingHTTPServer, dict[str, object]]:
    vec_idx_path, vec_meta_path = derive_vector_paths(index_path, vector_index_path, vector_metadata_path)
    loaded = load_vector_index(vec_idx_path, vec_meta_path)
    handler = _make_handler(
        loaded=loaded,
        default_embedding_model=embedding_model,
        default_nprobe=vector_nprobe,
        default_ef_search=vector_ef_search,
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    bind_host, bind_port = server.server_address
    info = {
        "ok": True,
        "host": bind_host,
        "port": int(bind_port),
        "index_path": str(Path(index_path).resolve()),
        "vector_index_path": str(Path(vec_idx_path).resolve()),
        "vector_metadata_path": str(Path(vec_meta_path).resolve()),
        "vector_index_type": str(getattr(loaded, "index_type", "flat")),
        "vector_shards": int(getattr(loaded, "shard_count", 1) or 1),
        "vector_rows": len(getattr(loaded, "snippet_ids", []) or []),
    }
    return server, info


def vector_service_serve(
    *,
    index_path: str,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_nprobe: int = 16,
    vector_ef_search: int = 64,
) -> dict[str, object]:
    server, info = start_vector_service_server(
        index_path=index_path,
        vector_index_path=vector_index_path,
        vector_metadata_path=vector_metadata_path,
        host=host,
        port=port,
        embedding_model=embedding_model,
        vector_nprobe=vector_nprobe,
        vector_ef_search=vector_ef_search,
    )
    stop_event = threading.Event()
    try:
        print(json.dumps(info, indent=2))
        while not stop_event.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return info
