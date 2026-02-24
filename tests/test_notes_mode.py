import json
from pathlib import Path

import pytest

from src.apps.notes_mode import run_notes_mode
from src.core.schemas import EvidenceItem, EvidencePack, ResearchReport


def _write_research_bundle(
    report_path: Path,
    include_key_claims: bool = True,
    include_synthesis_citation: bool = True,
) -> None:
    claim = "Boundary-aware distillation improves segmentation edges"
    sid = "Pcv_test:S1"
    cit = f"({sid})"

    synthesis = f"{claim} {cit}" if include_synthesis_citation else claim
    key_claim = f"{claim} {cit}" if include_synthesis_citation else claim
    rep = ResearchReport(
        question="q",
        shortlist=[],
        synthesis=synthesis,
        key_claims=[key_claim],
        gaps=[],
        experiments=[],
        citations=[cit] if include_synthesis_citation else [],
    )
    ev = EvidencePack(
        question="q",
        items=[EvidenceItem(paper_id="Pcv_test", snippet_id=sid, score=0.7, section="body", text=claim)],
    )

    lines = ["# Research Report", "", "## Synthesis", rep.synthesis, ""]
    if include_key_claims:
        lines.extend(["## Key Claims", f"- {claim} {cit}", ""])
    lines.extend(["## Citations", f"- {cit}", ""])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    report_path.with_suffix(".json").write_text(json.dumps(rep.model_dump(), indent=2), encoding="utf-8")
    report_path.with_suffix(".evidence.json").write_text(json.dumps(ev.model_dump(), indent=2), encoding="utf-8")
    report_path.with_suffix(".metrics.json").write_text(json.dumps({"citation_coverage": 1.0}, indent=2), encoding="utf-8")


def test_notes_mode_runs_research_then_ingests(tmp_path, monkeypatch) -> None:
    def _fake_run_research(**kwargs) -> str:
        out = Path(kwargs["out_path"])
        _write_research_bundle(out, include_key_claims=True)
        return str(out)

    def _fake_ingest(**_kwargs) -> dict:
        return {
            "kb_ingest_succeeded": True,
            "kb_claims_added": 2,
            "kb_claims_updated": 1,
            "kb_claims_disputed": 0,
        }

    monkeypatch.setattr("src.apps.notes_mode.run_research", _fake_run_research)
    monkeypatch.setattr("src.apps.notes_mode.ingest_report_to_kb", _fake_ingest)
    monkeypatch.chdir(tmp_path)

    out = tmp_path / "runs" / "notes" / "cv_notes.md"
    payload = run_notes_mode(
        question="Create study notes on boundary segmentation failures",
        index_path="data/indexes/bm25_index.json",
        kb_db="data/kb/knowledge.db",
        topic="computer_vision",
        out_path=str(out),
        sources_config_path="config/sources.yaml",
    )

    assert payload["mode"] == "notes"
    assert payload["kb_ingest_succeeded"] is True
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "# Study Notes" in text
    assert "## Key Claims" in text


def test_notes_mode_normalizes_missing_key_claims_from_synthesis(tmp_path, monkeypatch) -> None:
    def _fake_run_research(**kwargs) -> str:
        out = Path(kwargs["out_path"])
        _write_research_bundle(out, include_key_claims=False)
        return str(out)

    monkeypatch.setattr("src.apps.notes_mode.ingest_report_to_kb", lambda **_kwargs: {"kb_ingest_succeeded": True, "kb_claims_added": 1, "kb_claims_updated": 0, "kb_claims_disputed": 0})
    monkeypatch.setattr("src.apps.notes_mode.run_research", _fake_run_research)
    monkeypatch.chdir(tmp_path)

    out = tmp_path / "runs" / "notes" / "normalized.md"
    payload = run_notes_mode(
        question="Create notes",
        index_path="data/indexes/bm25_index.json",
        kb_db="data/kb/knowledge.db",
        topic="computer_vision",
        out_path=str(out),
    )
    assert payload["kb_ingest_succeeded"] is True
    assert out.exists()


def test_notes_mode_fails_without_any_cited_claims(tmp_path, monkeypatch) -> None:
    def _fake_run_research(**kwargs) -> str:
        out = Path(kwargs["out_path"])
        _write_research_bundle(out, include_key_claims=False, include_synthesis_citation=False)
        return str(out)

    monkeypatch.setattr("src.apps.notes_mode.run_research", _fake_run_research)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="missing cited key claims"):
        run_notes_mode(
            question="Create notes",
            index_path="data/indexes/bm25_index.json",
            kb_db="data/kb/knowledge.db",
            topic="computer_vision",
            out_path=str(tmp_path / "runs" / "notes" / "bad.md"),
        )
