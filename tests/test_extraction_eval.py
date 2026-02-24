from __future__ import annotations

from pathlib import Path
import json

from src.apps.extraction_eval import evaluate_extraction_against_gold, scaffold_extraction_gold, write_extraction_eval_report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_extraction_eval_scores_contains_and_ordered_checks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_json(
        corpus / "a.json",
        {
            "paper": {"paper_id": "arxiv:1"},
            "snippets": [
                {"text": "We propose a lightweight segmentation model for mobile."},
                {"text": "Limitation is hair boundary under low light. We add boundary loss."},
            ],
            "extraction_meta": {"page_stats": [{"page": 1, "chars": 120, "empty": False}]},
        },
    )

    gold = tmp_path / "gold.json"
    _write_json(
        gold,
        {
            "papers": [
                {
                    "paper_id": "arxiv:1",
                    "checks": [
                        {"id": "c1", "type": "contains", "needle": "lightweight segmentation", "weight": 1.0},
                        {
                            "id": "c2",
                            "type": "ordered_contains",
                            "needles": ["limitation", "boundary loss"],
                            "weight": 1.0,
                        },
                        {"id": "c3", "type": "page_nonempty_ratio", "min_ratio": 1.0, "weight": 1.0},
                    ],
                }
            ]
        },
    )

    report = evaluate_extraction_against_gold(str(corpus), str(gold))
    summary = report["summary"]
    assert summary["total_checks"] == 3
    assert summary["passed_checks"] == 3
    assert summary["weighted_score"] == 1.0


def test_extraction_eval_handles_missing_paper(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_json(corpus / "a.json", {"paper": {"paper_id": "exists"}, "snippets": [{"text": "x"}]})
    gold = tmp_path / "gold.json"
    _write_json(
        gold,
        {
            "papers": [
                {"paper_id": "exists", "checks": [{"id": "ok", "type": "contains", "needle": "x"}]},
                {"paper_id": "missing", "checks": [{"id": "m1", "type": "contains", "needle": "y"}]},
            ]
        },
    )

    report = evaluate_extraction_against_gold(str(corpus), str(gold))
    assert report["summary"]["total_checks"] == 2
    assert report["summary"]["passed_checks"] == 1
    missing = [r for r in report["results"] if r["paper_id"] == "missing"][0]
    assert missing["checks"][0]["detail"] == "paper_missing_in_corpus"


def test_extraction_eval_matches_by_doi_when_paper_id_differs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_json(
        corpus / "a.json",
        {
            "paper": {"paper_id": "openalex:W123", "doi": "10.1000/abc", "title": "A Finance Paper"},
            "snippets": [{"text": "alpha beta"}],
        },
    )
    gold = tmp_path / "gold.json"
    _write_json(
        gold,
        {
            "papers": [
                {
                    "paper_id": "doi:10.1000/abc",
                    "doi": "10.1000/abc",
                    "checks": [{"id": "ok", "type": "contains", "needle": "alpha"}],
                }
            ]
        },
    )
    report = evaluate_extraction_against_gold(str(corpus), str(gold))
    assert report["summary"]["papers_found_in_corpus"] == 1
    row = report["results"][0]
    assert row["matched_paper_id"] == "openalex:W123"
    assert row["checks"][0]["passed"] is True


def test_write_extraction_eval_report_outputs_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_json(corpus / "a.json", {"paper": {"paper_id": "p1"}, "snippets": [{"text": "abc"}]})
    gold = tmp_path / "gold.json"
    _write_json(gold, {"papers": [{"paper_id": "p1", "checks": [{"id": "c1", "type": "contains", "needle": "abc"}]}]})

    out_md, report = write_extraction_eval_report(str(corpus), str(gold), str(tmp_path / "eval.md"))
    assert Path(out_md).exists()
    assert Path(out_md).with_suffix(".json").exists()
    assert report["summary"]["weighted_score"] == 1.0


def test_scaffold_extraction_gold_creates_template(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _write_json(
        corpus / "a.json",
        {
            "paper": {"paper_id": "p1"},
            "snippets": [{"text": "alpha beta gamma delta epsilon zeta eta"}, {"text": "another snippet"}],
        },
    )
    out = tmp_path / "gold_generated.json"
    generated = scaffold_extraction_gold(str(corpus), str(out), max_papers=5, checks_per_paper=2, min_chars=300)
    payload = json.loads(Path(generated).read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["papers"][0]["paper_id"] == "p1"
    assert payload["papers"][0]["checks"][0]["type"] == "min_chars"
