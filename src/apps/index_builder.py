from __future__ import annotations

import sqlite3
import json
from pathlib import Path

from src.apps.corpus_health import evaluate_corpus_health
from src.apps.evidence_extract import load_snippets
from src.apps.retrieval import SimpleBM25Index, save_index
from src.apps.vector_index import DEFAULT_EMBEDDING_MODEL, build_vector_index, save_vector_index
from src.core.schemas import SnippetRecord


def _load_metadata_by_paper(db_path: str) -> dict[str, tuple[int | None, str | None, int]]:
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute("SELECT paper_id, year, venue, citation_count FROM papers").fetchall()
        except sqlite3.OperationalError:
            return {}
    out: dict[str, tuple[int | None, str | None, int]] = {}
    for paper_id, year, venue, citation_count in rows:
        out[str(paper_id)] = (year, venue, int(citation_count or 0))
    return out


def build_index(
    corpus_dir: str,
    out_path: str,
    db_path: str | None = None,
    with_vector: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_index_path: str | None = None,
    vector_metadata_path: str | None = None,
    batch_size: int = 64,
    vector_index_type: str = "flat",
    vector_nlist: int = 1024,
    vector_m: int = 32,
    vector_ef_construction: int = 200,
    vector_shards: int = 1,
    vector_train_sample_size: int = 200000,
    require_healthy_corpus: bool = False,
    min_snippets: int = 1,
    min_avg_chars_per_paper: int = 500,
    min_avg_chars_per_page: int = 80,
    max_extract_error_rate: float = 0.8,
    live_data_paths: list[str] | None = None,
) -> dict[str, object]:
    health_stats: dict[str, object] | None = None
    if require_healthy_corpus:
        health_stats = evaluate_corpus_health(
            corpus_dir=corpus_dir,
            min_snippets=min_snippets,
            min_avg_chars_per_paper=min_avg_chars_per_paper,
            min_avg_chars_per_page=min_avg_chars_per_page,
            max_extract_error_rate=max_extract_error_rate,
        )
        if not bool(health_stats.get("healthy", False)):
            reasons = ", ".join(str(r) for r in health_stats.get("reasons", []))
            warnings = ", ".join(str(w) for w in health_stats.get("warnings", []))
            detail = f"reasons=[{reasons}]"
            if warnings:
                detail += f"; warnings=[{warnings}]"
            raise ValueError(f"Corpus health check failed before build-index: {detail}")
    snippets = load_snippets(corpus_dir)
    live_added = 0
    for pth in live_data_paths or []:
        p = Path(pth)
        if not p.exists():
            raise ValueError(f"Live data file does not exist: {pth}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload.get("snippets", payload)
        if not isinstance(rows, list):
            raise ValueError(f"Live data file must contain a snippets list: {pth}")
        for row in rows:
            snippets.append(SnippetRecord(**row))
            live_added += 1
    if not snippets:
        raise ValueError(
            f"No snippets found in corpus '{corpus_dir}'. "
            "Run extract-corpus first and ensure it produced JSON snippet files."
        )
    enriched = 0
    if db_path:
        meta = _load_metadata_by_paper(db_path)
        for s in snippets:
            if s.paper_id not in meta:
                continue
            year, venue, cites = meta[s.paper_id]
            before = (s.paper_year, s.paper_venue, s.citation_count)
            if s.paper_year is None:
                s.paper_year = year
            if not s.paper_venue:
                s.paper_venue = venue
            if not s.citation_count:
                s.citation_count = cites
            after = (s.paper_year, s.paper_venue, s.citation_count)
            if before != after:
                enriched += 1
    index = SimpleBM25Index.build(snippets)
    save_index(index, out_path)
    stats: dict[str, object] = {
        "snippets": len(snippets),
        "enriched": enriched,
        "vector_enabled": False,
        "live_snippets_added": live_added,
    }
    if with_vector:
        from src.apps.retrieval import derive_vector_paths

        idx_path, meta_path = derive_vector_paths(out_path, vector_index_path, vector_metadata_path)
        bundle = build_vector_index(
            snippets,
            model_name=embedding_model,
            batch_size=batch_size,
            index_type=vector_index_type,
            nlist=vector_nlist,
            m=vector_m,
            ef_construction=vector_ef_construction,
            shards=vector_shards,
            train_sample_size=vector_train_sample_size,
        )
        save_vector_index(bundle, idx_path, meta_path)
        stats.update(
            {
                "vector_enabled": True,
                "vector_dim": bundle.dimension,
                "vector_rows": len(bundle.snippet_ids),
                "vector_index_type": bundle.index_type,
                "vector_shards": int(bundle.shard_count or 1),
                "vector_ann_params": bundle.ann_params or {},
                "vector_build_stats": bundle.build_stats or {},
            }
        )
    if health_stats is not None:
        stats["corpus_health"] = health_stats
    return stats
