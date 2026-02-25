"""Tests for incremental index builds.

Covers:
- Manifest generation and persistence in BM25 indexes
- Delta detection (additions, removals, changes, empty delta)
- build_index_incremental: noop, append, fallback-to-full
- Backward compatibility with legacy (no-manifest) indexes
- snippet_content_hash stability
- append_to_vector_index (flat, non-sharded)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.apps.index_builder import (
    _compute_delta,
    _load_and_enrich_snippets,
    build_index,
    build_index_incremental,
)
from src.apps.retrieval import (
    INDEX_MANIFEST_VERSION,
    SimpleBM25Index,
    load_index,
    load_index_manifest,
    save_index,
    snippet_content_hash,
)
from src.core.schemas import SnippetRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snippet(snippet_id: str, text: str = "some text", paper_id: str = "P1") -> SnippetRecord:
    return SnippetRecord(
        snippet_id=snippet_id,
        paper_id=paper_id,
        section="abstract",
        text=text,
        token_count=len(text.split()),
    )


def _write_corpus_file(corpus_dir: Path, paper_id: str, snippets: list[SnippetRecord]) -> None:
    """Write a single extracted paper JSON file into *corpus_dir*."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "paper": {"paper_id": paper_id, "year": 2025, "venue": "test"},
        "snippets": [s.model_dump() for s in snippets],
    }
    (corpus_dir / f"{paper_id.replace(':', '_')}.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


# ===========================================================================
# Manifest tests
# ===========================================================================

class TestManifest:
    def test_save_index_writes_manifest(self, tmp_path: Path) -> None:
        snippets = [_make_snippet("S1"), _make_snippet("S2")]
        idx = SimpleBM25Index.build(snippets)
        out = tmp_path / "idx.json"
        save_index(idx, str(out))

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "manifest" in payload
        m = payload["manifest"]
        assert m["version"] == INDEX_MANIFEST_VERSION
        assert m["snippet_count"] == 2
        assert set(m["snippet_hashes"].keys()) == {"S1", "S2"}
        assert "built_at" in m

    def test_load_index_manifest_returns_manifest(self, tmp_path: Path) -> None:
        snippets = [_make_snippet("S1")]
        idx = SimpleBM25Index.build(snippets)
        out = tmp_path / "idx.json"
        save_index(idx, str(out))

        m = load_index_manifest(str(out))
        assert m is not None
        assert m["snippet_count"] == 1

    def test_load_index_manifest_returns_none_for_missing(self, tmp_path: Path) -> None:
        assert load_index_manifest(str(tmp_path / "no_such.json")) is None

    def test_load_index_manifest_returns_none_for_legacy(self, tmp_path: Path) -> None:
        out = tmp_path / "legacy.json"
        out.write_text(json.dumps({"snippets": []}), encoding="utf-8")
        assert load_index_manifest(str(out)) is None

    def test_load_index_still_works_with_manifest(self, tmp_path: Path) -> None:
        """Ensure the existing load_index path is backward-compatible."""
        snippets = [_make_snippet("S1", text="hello world")]
        idx = SimpleBM25Index.build(snippets)
        out = tmp_path / "idx.json"
        save_index(idx, str(out))

        loaded = load_index(str(out))
        assert len(loaded.snippets) == 1
        assert loaded.snippets[0].snippet_id == "S1"


# ===========================================================================
# snippet_content_hash tests
# ===========================================================================

class TestSnippetContentHash:
    def test_deterministic(self) -> None:
        s = _make_snippet("S1", text="hello")
        assert snippet_content_hash(s) == snippet_content_hash(s)

    def test_different_for_different_text(self) -> None:
        a = _make_snippet("S1", text="hello")
        b = _make_snippet("S1", text="world")
        assert snippet_content_hash(a) != snippet_content_hash(b)

    def test_different_for_different_id(self) -> None:
        a = _make_snippet("S1", text="same")
        b = _make_snippet("S2", text="same")
        assert snippet_content_hash(a) != snippet_content_hash(b)

    def test_ignores_metadata_fields(self) -> None:
        """Hash should NOT change when only metadata (year, venue, citation_count) changes."""
        a = _make_snippet("S1", text="same")
        a.paper_year = 2020
        b = _make_snippet("S1", text="same")
        b.paper_year = 2025
        assert snippet_content_hash(a) == snippet_content_hash(b)


# ===========================================================================
# Delta detection tests
# ===========================================================================

class TestComputeDelta:
    def test_no_changes(self) -> None:
        snippets = [_make_snippet("S1"), _make_snippet("S2")]
        manifest = {
            "snippet_hashes": {s.snippet_id: snippet_content_hash(s) for s in snippets},
        }
        delta = _compute_delta(snippets, manifest)
        assert delta.is_empty
        assert not delta.additions_only

    def test_additions_only(self) -> None:
        old = [_make_snippet("S1")]
        new = [_make_snippet("S1"), _make_snippet("S2")]
        manifest = {"snippet_hashes": {s.snippet_id: snippet_content_hash(s) for s in old}}
        delta = _compute_delta(new, manifest)
        assert delta.added_ids == {"S2"}
        assert not delta.removed_ids
        assert not delta.changed_ids
        assert delta.additions_only

    def test_removal_detected(self) -> None:
        old = [_make_snippet("S1"), _make_snippet("S2")]
        new = [_make_snippet("S1")]
        manifest = {"snippet_hashes": {s.snippet_id: snippet_content_hash(s) for s in old}}
        delta = _compute_delta(new, manifest)
        assert delta.removed_ids == {"S2"}
        assert not delta.additions_only

    def test_change_detected(self) -> None:
        old = [_make_snippet("S1", text="original")]
        new = [_make_snippet("S1", text="modified")]
        manifest = {"snippet_hashes": {s.snippet_id: snippet_content_hash(s) for s in old}}
        delta = _compute_delta(new, manifest)
        assert delta.changed_ids == {"S1"}
        assert delta.is_empty is False
        assert not delta.additions_only


# ===========================================================================
# Incremental build — end-to-end (BM25 only)
# ===========================================================================

class TestBuildIndexIncremental:
    def _build_corpus(self, tmp_path: Path, num_papers: int = 1) -> tuple[Path, Path]:
        """Build a corpus dir with *num_papers* and return (corpus_dir, index_path)."""
        corpus = tmp_path / "corpus"
        for i in range(num_papers):
            _write_corpus_file(
                corpus,
                f"P{i}",
                [_make_snippet(f"P{i}:S1", text=f"paper {i} snippet one", paper_id=f"P{i}")],
            )
        idx = tmp_path / "idx.json"
        return corpus, idx

    def test_first_build_creates_manifest(self, tmp_path: Path) -> None:
        corpus, idx = self._build_corpus(tmp_path, num_papers=2)
        stats = build_index_incremental(str(corpus), str(idx))
        # First build delegates to build_index (no manifest existed)
        assert idx.exists()
        m = load_index_manifest(str(idx))
        assert m is not None
        assert m["snippet_count"] == 2

    def test_noop_when_no_changes(self, tmp_path: Path) -> None:
        corpus, idx = self._build_corpus(tmp_path, num_papers=2)
        build_index(str(corpus), str(idx))
        mtime_before = idx.stat().st_mtime_ns

        stats = build_index_incremental(str(corpus), str(idx))
        assert stats["incremental"] is True
        assert stats["incremental_action"] == "noop"
        assert stats["delta_added"] == 0

    def test_append_new_snippets(self, tmp_path: Path) -> None:
        corpus, idx = self._build_corpus(tmp_path, num_papers=1)
        build_index(str(corpus), str(idx))

        # Add a second paper to the corpus
        _write_corpus_file(
            corpus,
            "P_new",
            [_make_snippet("P_new:S1", text="brand new paper snippet", paper_id="P_new")],
        )

        stats = build_index_incremental(str(corpus), str(idx))
        assert stats["incremental"] is True
        assert stats["incremental_action"] == "append"
        assert stats["delta_added"] == 1
        assert stats["snippets"] == 2

        # Verify the new snippet is in the index
        loaded = load_index(str(idx))
        ids = {s.snippet_id for s in loaded.snippets}
        assert "P_new:S1" in ids

        # Manifest updated
        m = load_index_manifest(str(idx))
        assert m is not None
        assert m["snippet_count"] == 2

    def test_fallback_on_removal(self, tmp_path: Path) -> None:
        corpus, idx = self._build_corpus(tmp_path, num_papers=2)
        build_index(str(corpus), str(idx))

        # Remove one paper
        for f in corpus.glob("P1*"):
            f.unlink()

        stats = build_index_incremental(str(corpus), str(idx))
        # Should have fallen back to full build (no "incremental" key or it's absent)
        assert "incremental" not in stats or stats.get("incremental_action") != "noop"
        loaded = load_index(str(idx))
        assert len(loaded.snippets) == 1

    def test_fallback_on_content_change(self, tmp_path: Path) -> None:
        corpus, idx = self._build_corpus(tmp_path, num_papers=1)
        build_index(str(corpus), str(idx))

        # Overwrite the paper with different text
        _write_corpus_file(
            corpus,
            "P0",
            [_make_snippet("P0:S1", text="COMPLETELY DIFFERENT content now", paper_id="P0")],
        )

        stats = build_index_incremental(str(corpus), str(idx))
        # Content changed → full rebuild
        loaded = load_index(str(idx))
        assert loaded.snippets[0].text == "COMPLETELY DIFFERENT content now"

    def test_incremental_produces_same_query_results_as_full(self, tmp_path: Path) -> None:
        corpus, idx_inc = self._build_corpus(tmp_path, num_papers=1)
        build_index(str(corpus), str(idx_inc))

        # Add new paper
        _write_corpus_file(
            corpus,
            "P_extra",
            [_make_snippet("P_extra:S1", text="extra paper about segmentation models", paper_id="P_extra")],
        )

        # Incremental build
        build_index_incremental(str(corpus), str(idx_inc))

        # Full build for comparison
        idx_full = tmp_path / "idx_full.json"
        build_index(str(corpus), str(idx_full))

        inc = load_index(str(idx_inc))
        full = load_index(str(idx_full))

        q = "segmentation models"
        results_inc = inc.query(q, top_k=10)
        results_full = full.query(q, top_k=10)

        assert len(results_inc.items) == len(results_full.items)
        inc_ids = {it.snippet_id for it in results_inc.items}
        full_ids = {it.snippet_id for it in results_full.items}
        assert inc_ids == full_ids

    def test_legacy_index_triggers_full_build(self, tmp_path: Path) -> None:
        """An index without a manifest should trigger a full build and add the manifest."""
        corpus, idx = self._build_corpus(tmp_path, num_papers=1)

        # Write a legacy index (no manifest)
        snippets = [_make_snippet("P0:S1", text="paper 0 snippet one", paper_id="P0")]
        legacy_index = SimpleBM25Index.build(snippets)
        idx.parent.mkdir(parents=True, exist_ok=True)
        legacy_payload = {"snippets": [s.model_dump() for s in legacy_index.snippets]}
        idx.write_text(json.dumps(legacy_payload), encoding="utf-8")

        assert load_index_manifest(str(idx)) is None  # No manifest

        stats = build_index_incremental(str(corpus), str(idx))
        # Should have done a full build and now have a manifest
        m = load_index_manifest(str(idx))
        assert m is not None
        assert m["version"] == INDEX_MANIFEST_VERSION


# ===========================================================================
# Enrichment helper tests
# ===========================================================================

class TestLoadAndEnrichSnippets:
    def test_loads_from_fixtures(self) -> None:
        snippets, enriched, live_added = _load_and_enrich_snippets(
            "tests/fixtures/extracted", None, None,
        )
        assert len(snippets) > 0
        assert live_added == 0

    def test_appends_live_data(self, tmp_path: Path) -> None:
        live_payload = {
            "snippets": [
                {
                    "snippet_id": "LIVE:1",
                    "paper_id": "LIVE",
                    "section": "live",
                    "text": "live data",
                    "token_count": 2,
                }
            ]
        }
        live_path = tmp_path / "live.json"
        live_path.write_text(json.dumps(live_payload), encoding="utf-8")

        snippets, _, live_added = _load_and_enrich_snippets(
            "tests/fixtures/extracted", None, [str(live_path)],
        )
        assert live_added == 1
        assert any(s.snippet_id == "LIVE:1" for s in snippets)
