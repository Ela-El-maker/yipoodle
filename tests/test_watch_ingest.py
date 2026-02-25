from __future__ import annotations

from pathlib import Path

from src.apps.watch_ingest import run_watch_ingest


def test_watch_ingest_once_processes_existing_pdf(tmp_path, monkeypatch) -> None:
    watch_dir = tmp_path / "papers"
    watch_dir.mkdir(parents=True)
    (watch_dir / "a.pdf").write_bytes(b"%PDF-1.4\n")

    calls = {"extract": 0, "index": 0}

    def _fake_extract(*, papers_dir, out_dir, db_path, min_text_chars):  # noqa: ANN001
        calls["extract"] += 1
        return {"processed": 1, "created": 1}

    def _fake_index(*, corpus_dir, out_path, db_path):  # noqa: ANN001
        calls["index"] += 1
        return {"snippets": 3}

    monkeypatch.setattr("src.apps.watch_ingest.extract_from_papers_dir_with_db", _fake_extract)
    monkeypatch.setattr("src.apps.watch_ingest.build_index_incremental", _fake_index)

    out = run_watch_ingest(
        watch_dir=str(watch_dir),
        extracted_dir=str(tmp_path / "extracted"),
        db_path=str(tmp_path / "papers.db"),
        index_path=str(tmp_path / "idx.json"),
        once=True,
        out_path=str(tmp_path / "watch.json"),
    )
    assert out["processed_count"] == 1
    assert calls["extract"] == 1
    assert calls["index"] == 1
    assert Path(tmp_path / "watch.json").exists()
