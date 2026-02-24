from __future__ import annotations

import logging
import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.apps.corpus_health import evaluate_corpus_health
from src.apps.evidence_extract import load_snippets
from src.apps.retrieval import (
    SimpleBM25Index,
    load_index_manifest,
    save_index,
    snippet_content_hash,
)
from src.apps.vector_index import DEFAULT_EMBEDDING_MODEL, build_vector_index, save_vector_index
from src.core.schemas import SnippetRecord

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Snippet loading & enrichment helpers (shared by full and incremental builds)
# ---------------------------------------------------------------------------

def _load_and_enrich_snippets(
    corpus_dir: str,
    db_path: str | None,
    live_data_paths: list[str] | None,
) -> tuple[list[SnippetRecord], int, int]:
    """Load snippets from *corpus_dir*, enrich from DB, append live data.

    Returns ``(snippets, enriched_count, live_added_count)``.
    """
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
    return snippets, enriched, live_added


# ---------------------------------------------------------------------------
# Delta detection
# ---------------------------------------------------------------------------

@dataclass
class _IndexDelta:
    """Result of comparing current corpus snippets against an existing index manifest."""

    added_ids: set[str]
    removed_ids: set[str]
    changed_ids: set[str]

    @property
    def additions_only(self) -> bool:
        return bool(self.added_ids) and not self.removed_ids and not self.changed_ids

    @property
    def is_empty(self) -> bool:
        return not self.added_ids and not self.removed_ids and not self.changed_ids


def _compute_delta(
    current_snippets: list[SnippetRecord],
    manifest: dict[str, Any],
) -> _IndexDelta:
    """Compare *current_snippets* against the hashes stored in *manifest*."""
    old_hashes: dict[str, str] = manifest.get("snippet_hashes", {})
    new_hashes = {s.snippet_id: snippet_content_hash(s) for s in current_snippets}

    old_ids = set(old_hashes)
    new_ids = set(new_hashes)

    added = new_ids - old_ids
    removed = old_ids - new_ids
    changed = {sid for sid in (old_ids & new_ids) if old_hashes[sid] != new_hashes[sid]}
    return _IndexDelta(added_ids=added, removed_ids=removed, changed_ids=changed)


# ---------------------------------------------------------------------------
# Incremental build
# ---------------------------------------------------------------------------

def build_index_incremental(
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
    """Build the index incrementally when possible.

    Logic:
    1. Load existing manifest from *out_path*.
    2. If no manifest exists (legacy index or first build) → full rebuild.
    3. Compute delta between current corpus and manifest.
    4. If delta is empty → no-op, return immediately.
    5. If additions only → append new snippets to existing index.
    6. If removals or changes → full rebuild (safe fallback).

    For the vector index, additions-only with ``flat`` index type can be
    appended via ``faiss.add()``.  Other cases trigger a full vector rebuild.
    """
    manifest = load_index_manifest(out_path)
    if manifest is None:
        log.info("No existing manifest found at %s — performing full build", out_path)
        return build_index(
            corpus_dir=corpus_dir,
            out_path=out_path,
            db_path=db_path,
            with_vector=with_vector,
            embedding_model=embedding_model,
            vector_index_path=vector_index_path,
            vector_metadata_path=vector_metadata_path,
            batch_size=batch_size,
            vector_index_type=vector_index_type,
            vector_nlist=vector_nlist,
            vector_m=vector_m,
            vector_ef_construction=vector_ef_construction,
            vector_shards=vector_shards,
            vector_train_sample_size=vector_train_sample_size,
            require_healthy_corpus=require_healthy_corpus,
            min_snippets=min_snippets,
            min_avg_chars_per_paper=min_avg_chars_per_paper,
            min_avg_chars_per_page=min_avg_chars_per_page,
            max_extract_error_rate=max_extract_error_rate,
            live_data_paths=live_data_paths,
        )

    # ---- Health gate (same as full build) ----
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

    snippets, enriched, live_added = _load_and_enrich_snippets(corpus_dir, db_path, live_data_paths)
    if not snippets:
        raise ValueError(
            f"No snippets found in corpus '{corpus_dir}'. "
            "Run extract-corpus first and ensure it produced JSON snippet files."
        )

    delta = _compute_delta(snippets, manifest)

    if delta.is_empty:
        log.info("Index is up-to-date — no changes detected (skipping rebuild)")
        stats: dict[str, object] = {
            "snippets": len(snippets),
            "enriched": enriched,
            "vector_enabled": False,
            "live_snippets_added": live_added,
            "incremental": True,
            "incremental_action": "noop",
            "delta_added": 0,
            "delta_removed": 0,
            "delta_changed": 0,
        }
        if health_stats is not None:
            stats["corpus_health"] = health_stats
        return stats

    if not delta.additions_only:
        log.info(
            "Delta contains removals (%d) or changes (%d) — falling back to full rebuild",
            len(delta.removed_ids),
            len(delta.changed_ids),
        )
        return build_index(
            corpus_dir=corpus_dir,
            out_path=out_path,
            db_path=db_path,
            with_vector=with_vector,
            embedding_model=embedding_model,
            vector_index_path=vector_index_path,
            vector_metadata_path=vector_metadata_path,
            batch_size=batch_size,
            vector_index_type=vector_index_type,
            vector_nlist=vector_nlist,
            vector_m=vector_m,
            vector_ef_construction=vector_ef_construction,
            vector_shards=vector_shards,
            vector_train_sample_size=vector_train_sample_size,
            require_healthy_corpus=False,   # already checked above
            live_data_paths=live_data_paths,
        )

    # ---- Additions only → incremental append ----
    log.info("Incremental update: appending %d new snippets", len(delta.added_ids))

    # Full BM25 rebuild is cheap (pure Python, no embeddings) so we always
    # rebuild it from the complete snippet list to keep IDF accurate.
    index = SimpleBM25Index.build(snippets)
    save_index(index, out_path)

    stats = {
        "snippets": len(snippets),
        "enriched": enriched,
        "vector_enabled": False,
        "live_snippets_added": live_added,
        "incremental": True,
        "incremental_action": "append",
        "delta_added": len(delta.added_ids),
        "delta_removed": 0,
        "delta_changed": 0,
    }

    if with_vector:
        from src.apps.retrieval import derive_vector_paths
        from src.apps.vector_index import (
            append_to_vector_index,
            load_vector_index,
        )

        idx_path, meta_path = derive_vector_paths(out_path, vector_index_path, vector_metadata_path)

        new_snippets = [s for s in snippets if s.snippet_id in delta.added_ids]

        can_append = (
            vector_index_type == "flat"
            and vector_shards <= 1
            and Path(meta_path).exists()
        )
        if can_append:
            try:
                vec_stats = append_to_vector_index(
                    new_snippets=new_snippets,
                    faiss_path=idx_path,
                    meta_path=meta_path,
                    model_name=embedding_model,
                    batch_size=batch_size,
                )
                stats.update({
                    "vector_enabled": True,
                    "vector_incremental": True,
                    **vec_stats,
                })
            except Exception:
                log.warning("Vector append failed — falling back to full vector rebuild")
                can_append = False

        if not can_append:
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
            stats.update({
                "vector_enabled": True,
                "vector_incremental": False,
                "vector_dim": bundle.dimension,
                "vector_rows": len(bundle.snippet_ids),
                "vector_index_type": bundle.index_type,
                "vector_shards": int(bundle.shard_count or 1),
                "vector_ann_params": bundle.ann_params or {},
                "vector_build_stats": bundle.build_stats or {},
            })

    if health_stats is not None:
        stats["corpus_health"] = health_stats
    return stats


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

    snippets, enriched, live_added = _load_and_enrich_snippets(corpus_dir, db_path, live_data_paths)
    if not snippets:
        raise ValueError(
            f"No snippets found in corpus '{corpus_dir}'. "
            "Run extract-corpus first and ensure it produced JSON snippet files."
        )
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
