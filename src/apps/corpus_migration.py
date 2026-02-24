from __future__ import annotations

from pathlib import Path
import json

from src.apps.evidence_extract import compute_extraction_quality


def _compact_text_from_snippets(snippets: list[dict]) -> str:
    return "\n".join(str(s.get("text", "")).strip() for s in snippets if str(s.get("text", "")).strip())


def migrate_extraction_meta(corpus_dir: str, dry_run: bool = False) -> dict[str, object]:
    files = sorted(Path(corpus_dir).glob("*.json"))
    scanned = 0
    updated = 0
    skipped = 0
    errors = 0

    for fp in files:
        scanned += 1
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            snippets = payload.get("snippets", [])
            existing = payload.get("extraction_meta")
            if existing and all(
                k in existing
                for k in ("extractor", "quality_score", "quality_band", "pages_total", "empty_pages", "page_stats")
            ):
                skipped += 1
                continue

            text = _compact_text_from_snippets(snippets)
            quality_score, quality_band, _signals = compute_extraction_quality(text)
            ocr_applied = any(str(s.get("extraction_source", "")).lower() == "ocr" for s in snippets)
            extraction_meta = {
                "extractor": "legacy_unknown",
                "two_column_applied": False,
                "ocr_applied": bool(ocr_applied),
                "pages_total": 0,
                "empty_pages": 0,
                "empty_page_pct": 0.0,
                "page_stats": [],
                "quality_score": round(float(quality_score), 4),
                "quality_band": quality_band,
                "migration_note": "backfilled_from_legacy_snippets",
            }
            payload["extraction_meta"] = extraction_meta
            if not dry_run:
                fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            updated += 1
        except Exception:
            errors += 1

    return {
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
    }
