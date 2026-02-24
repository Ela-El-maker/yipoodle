from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import json
import time

from src.core.schemas import SnippetRecord
from src.apps.vector_backend import (
    VectorBuildArtifact,
    VectorBuildConfig,
    VectorQueryConfig,
    get_faiss_backend,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorIndexBundle:
    index: object
    snippet_ids: list[str]
    embedding_model: str
    dimension: int
    index_type: str = "flat"
    ann_params: dict[str, int | float] | None = None
    shard_count: int = 1
    build_stats: dict[str, int | float] | None = None


@dataclass
class LoadedVectorIndex:
    index: object
    snippet_ids: list[str]
    embedding_model: str
    dimension: int
    metric: str
    index_mtime_ns: int
    index_type: str = "flat"
    ann_params: dict[str, int | float] | None = None
    shard_count: int = 1
    shard_paths: list[str] | None = None


def _import_vector_deps() -> tuple[object, object]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("FAISS unavailable. Install faiss-cpu or use --retrieval-mode lexical.") from exc
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "sentence-transformers unavailable. Install dependencies or use --retrieval-mode lexical."
        ) from exc
    return faiss, SentenceTransformer


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str) -> object:
    _, SentenceTransformer = _import_vector_deps()
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            f"Embedding model '{model_name}' unavailable locally. Pre-download it or enable network for first use."
        ) from exc


def _encode_texts(model_name: str, texts: list[str], batch_size: int = 64):
    model = _get_embedding_model(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs


def _build_one_index(
    embeddings,
    snippet_ids: list[str],
    model_name: str,
    index_type: str,
    nlist: int,
    m: int,
    ef_construction: int,
    train_sample_size: int,
) -> VectorBuildArtifact:
    faiss, _ = _import_vector_deps()
    backend = get_faiss_backend(faiss, index_type=index_type)
    cfg = VectorBuildConfig(
        model_name=model_name,
        index_type=index_type,
        nlist=nlist,
        m=m,
        ef_construction=ef_construction,
        train_sample_size=train_sample_size,
    )
    return backend.build(embeddings, snippet_ids, cfg)


def _shard_bucket(sid: str, shards: int) -> int:
    if shards <= 1:
        return 0
    # Stable across processes/runs (unlike Python's built-in hash randomization).
    digest = hashlib.sha1(sid.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return value % shards


def build_vector_index(
    snippets: list[SnippetRecord],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    index_type: str = "flat",
    nlist: int = 1024,
    m: int = 32,
    ef_construction: int = 200,
    shards: int = 1,
    train_sample_size: int = 200000,
) -> VectorIndexBundle:
    if not snippets:
        raise ValueError("Cannot build vector index with zero snippets")
    texts = [s.text for s in snippets]
    snippet_ids = [s.snippet_id for s in snippets]
    embeddings = _encode_texts(model_name, texts, batch_size=batch_size)

    t0 = time.perf_counter()
    if int(shards) <= 1:
        art = _build_one_index(
            embeddings,
            snippet_ids,
            model_name=model_name,
            index_type=index_type,
            nlist=nlist,
            m=m,
            ef_construction=ef_construction,
            train_sample_size=train_sample_size,
        )
        build_ms = (time.perf_counter() - t0) * 1000.0
        stats = dict(art.build_stats or {})
        stats.update({"build_time_ms": round(build_ms, 3), "memory_estimate_mb": round((embeddings.nbytes / 1024 / 1024), 3)})
        return VectorIndexBundle(
            index=art.index,
            snippet_ids=art.snippet_ids,
            embedding_model=art.embedding_model,
            dimension=art.dimension,
            index_type=art.index_type,
            ann_params=art.ann_params,
            shard_count=1,
            build_stats=stats,
        )

    shard_count = max(1, int(shards))
    shard_rows: dict[int, list[int]] = {i: [] for i in range(shard_count)}
    for row, sid in enumerate(snippet_ids):
        shard_rows[_shard_bucket(sid, shard_count)].append(row)

    shard_indexes: list[object] = []
    merged_ids: list[str] = []
    build_stats: dict[str, int | float] = {"train_size": 0}
    for i in range(shard_count):
        rows = shard_rows.get(i, [])
        if not rows:
            shard_indexes.append(None)
            continue
        shard_emb = embeddings[rows]
        shard_ids = [snippet_ids[r] for r in rows]
        art = _build_one_index(
            shard_emb,
            shard_ids,
            model_name=model_name,
            index_type=index_type,
            nlist=nlist,
            m=m,
            ef_construction=ef_construction,
            train_sample_size=train_sample_size,
        )
        shard_indexes.append(art.index)
        merged_ids.extend(shard_ids)
        build_stats["train_size"] = int(build_stats.get("train_size", 0)) + int((art.build_stats or {}).get("train_size", 0))

    build_ms = (time.perf_counter() - t0) * 1000.0
    build_stats.update({"build_time_ms": round(build_ms, 3), "memory_estimate_mb": round((embeddings.nbytes / 1024 / 1024), 3)})

    dim = int(embeddings.shape[1])
    return VectorIndexBundle(
        index=shard_indexes,
        snippet_ids=merged_ids,
        embedding_model=model_name,
        dimension=dim,
        index_type=index_type,
        ann_params={"nlist": int(nlist), "m": int(m), "ef_construction": int(ef_construction)},
        shard_count=shard_count,
        build_stats=build_stats,
    )


def save_vector_index(bundle: VectorIndexBundle, faiss_path: str, meta_path: str) -> None:
    faiss, _ = _import_vector_deps()
    out_idx = Path(faiss_path)
    out_meta = Path(meta_path)
    out_idx.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str] = []
    if int(bundle.shard_count or 1) <= 1:
        faiss.write_index(bundle.index, str(out_idx))
        index_mtime_ns = out_idx.stat().st_mtime_ns
    else:
        indexes = list(bundle.index) if isinstance(bundle.index, list) else [bundle.index]
        for i, idx in enumerate(indexes):
            if idx is None:
                continue
            sp = out_idx.with_suffix(f".shard{i}.faiss")
            faiss.write_index(idx, str(sp))
            shard_paths.append(str(sp))
        index_mtime_ns = max((Path(p).stat().st_mtime_ns for p in shard_paths), default=0)

    meta = {
        "version": 2,
        "embedding_model": bundle.embedding_model,
        "dimension": bundle.dimension,
        "metric": "cosine_via_ip",
        "snippet_ids": bundle.snippet_ids,
        "snippet_count": len(bundle.snippet_ids),
        "index_mtime_ns": index_mtime_ns,
        "index_type": bundle.index_type or "flat",
        "ann_params": bundle.ann_params or {},
        "query_defaults": {"nprobe": 16, "ef_search": 64},
        "shard_count": int(bundle.shard_count or 1),
        "shard_paths": shard_paths,
        "build_stats": bundle.build_stats or {},
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_vector_index(faiss_path: str, meta_path: str) -> LoadedVectorIndex:
    faiss, _ = _import_vector_deps()
    p_idx = Path(faiss_path)
    p_meta = Path(meta_path)
    if not p_meta.exists():
        raise FileNotFoundError("Vector index artifacts not found; run build-index --with-vector or use --retrieval-mode lexical")
    meta = json.loads(p_meta.read_text(encoding="utf-8"))
    version = int(meta.get("version", 1))

    if version <= 1:
        if not p_idx.exists():
            raise FileNotFoundError("Vector index artifacts not found; run build-index --with-vector or use --retrieval-mode lexical")
        index = faiss.read_index(str(p_idx))
        snippet_ids = list(meta.get("snippet_ids", []))
        snippet_count = int(meta.get("snippet_count", 0))
        if snippet_count != len(snippet_ids) or snippet_count != int(index.ntotal):
            raise ValueError("Vector metadata mismatch: snippet_count/snippet_ids/faiss rows are inconsistent")
        return LoadedVectorIndex(
            index=index,
            snippet_ids=snippet_ids,
            embedding_model=str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
            dimension=int(meta.get("dimension", 0)),
            metric=str(meta.get("metric", "cosine_via_ip")),
            index_mtime_ns=int(meta.get("index_mtime_ns", 0)),
            index_type="flat",
            ann_params={},
            shard_count=1,
            shard_paths=[],
        )

    shard_count = int(meta.get("shard_count", 1) or 1)
    shard_paths = list(meta.get("shard_paths", []) or [])
    snippet_ids = list(meta.get("snippet_ids", []))
    snippet_count = int(meta.get("snippet_count", len(snippet_ids)))
    if snippet_count != len(snippet_ids):
        raise ValueError("Vector metadata mismatch: snippet_count/snippet_ids are inconsistent")

    if shard_count <= 1:
        if not p_idx.exists():
            raise FileNotFoundError("Vector index file missing for non-sharded metadata")
        index = faiss.read_index(str(p_idx))
        if int(index.ntotal) != snippet_count:
            raise ValueError("Vector metadata mismatch: faiss rows are inconsistent")
        return LoadedVectorIndex(
            index=index,
            snippet_ids=snippet_ids,
            embedding_model=str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
            dimension=int(meta.get("dimension", 0)),
            metric=str(meta.get("metric", "cosine_via_ip")),
            index_mtime_ns=int(meta.get("index_mtime_ns", 0)),
            index_type=str(meta.get("index_type", "flat")),
            ann_params=dict(meta.get("ann_params", {}) or {}),
            shard_count=1,
            shard_paths=[],
        )

    if not shard_paths:
        # recover conventional naming if paths missing
        shard_paths = [str(p_idx.with_suffix(f".shard{i}.faiss")) for i in range(shard_count)]

    indexes = []
    for sp in shard_paths:
        p = Path(sp)
        if not p.exists():
            raise FileNotFoundError(f"Missing shard file: {p}")
        indexes.append(faiss.read_index(str(p)))
    rows = sum(int(ix.ntotal) for ix in indexes)
    if rows != snippet_count:
        raise ValueError("Vector metadata mismatch: shard rows/snippet_ids are inconsistent")

    return LoadedVectorIndex(
        index=indexes,
        snippet_ids=snippet_ids,
        embedding_model=str(meta.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
        dimension=int(meta.get("dimension", 0)),
        metric=str(meta.get("metric", "cosine_via_ip")),
        index_mtime_ns=int(meta.get("index_mtime_ns", 0)),
        index_type=str(meta.get("index_type", "flat")),
        ann_params=dict(meta.get("ann_params", {}) or {}),
        shard_count=shard_count,
        shard_paths=shard_paths,
    )


def _query_single_index(loaded: LoadedVectorIndex, question_vec, top_k: int, nprobe: int = 16, ef_search: int = 64) -> list[tuple[int, float]]:
    faiss, _ = _import_vector_deps()
    backend = get_faiss_backend(faiss, loaded.index_type)
    cfg = VectorQueryConfig(top_k=top_k, nprobe=nprobe, ef_search=ef_search)
    return backend.query(loaded.index, question_vec, cfg)


def _query_sharded(loaded: LoadedVectorIndex, question_vec, top_k: int, nprobe: int = 16, ef_search: int = 64) -> list[tuple[int, float]]:
    # snippet_ids are stored as merged shard order; maintain running offset
    if not isinstance(loaded.index, list):
        return _query_single_index(loaded, question_vec, top_k=top_k, nprobe=nprobe, ef_search=ef_search)

    faiss, _ = _import_vector_deps()
    backend = get_faiss_backend(faiss, loaded.index_type)
    cfg = VectorQueryConfig(top_k=max(top_k, 1), nprobe=nprobe, ef_search=ef_search)

    merged: list[tuple[int, float]] = []
    offset = 0
    for ix in loaded.index:
        shard_rows = int(ix.ntotal)
        pairs = backend.query(ix, question_vec, cfg)
        for idx, score in pairs:
            if idx < 0:
                continue
            merged.append((offset + int(idx), float(score)))
        offset += shard_rows
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def query_vector_index(
    loaded: LoadedVectorIndex,
    question: str,
    top_k: int,
    model_name_override: str | None = None,
    nprobe: int = 16,
    ef_search: int = 64,
) -> list[tuple[str, float]]:
    if top_k <= 0:
        return []
    model_name = model_name_override or loaded.embedding_model
    q = _encode_texts(model_name, [question], batch_size=1).astype("float32")
    k = min(top_k, len(loaded.snippet_ids))
    if k == 0:
        return []

    if int(loaded.shard_count or 1) > 1:
        raw = _query_sharded(loaded, q, top_k=k, nprobe=nprobe, ef_search=ef_search)
    else:
        raw = _query_single_index(loaded, q, top_k=k, nprobe=nprobe, ef_search=ef_search)

    out: list[tuple[str, float]] = []
    for idx, score in raw:
        if idx < 0 or idx >= len(loaded.snippet_ids):
            continue
        out.append((loaded.snippet_ids[idx], float(score)))
    return out
