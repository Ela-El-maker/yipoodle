from pathlib import Path
import json

from src.apps.corpus_migration import migrate_extraction_meta


def test_migrate_extraction_meta_updates_legacy_file(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = {
        "paper": {"paper_id": "p1"},
        "snippets": [{"text": "legacy snippet text"}, {"text": "more text"}],
    }
    p = corpus / "p1.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    stats = migrate_extraction_meta(str(corpus), dry_run=False)
    assert stats["scanned"] == 1
    assert stats["updated"] == 1
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "extraction_meta" in data
    assert data["extraction_meta"]["migration_note"] == "backfilled_from_legacy_snippets"


def test_migrate_extraction_meta_dry_run(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    p = corpus / "p1.json"
    p.write_text(json.dumps({"paper": {"paper_id": "p1"}, "snippets": [{"text": "abc"}]}), encoding="utf-8")
    stats = migrate_extraction_meta(str(corpus), dry_run=True)
    assert stats["updated"] == 1
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "extraction_meta" not in data
