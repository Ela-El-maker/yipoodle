from __future__ import annotations

from pathlib import Path
import json

from src.apps.research_template import run_research_template


def test_run_research_template_executes_questions(tmp_path, monkeypatch) -> None:
    templates = tmp_path / "templates.yaml"
    templates.write_text(
        """
templates:
  lit_review:
    description: test
    questions:
      - "Q1 about {topic}"
      - "Q2 about {topic}"
""".strip(),
        encoding="utf-8",
    )

    calls: list[str] = []

    def _fake_research(*, question, out_path, **kwargs):  # noqa: ANN001
        calls.append(question)
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Report\n", encoding="utf-8")
        (p.with_suffix(".json")).write_text(
            json.dumps(
                {
                    "question": question,
                    "shortlist": [],
                    "synthesis": "S",
                    "key_claims": ["c1"],
                    "gaps": ["g1"],
                    "experiments": [],
                    "citations": ["P1:S1"],
                }
            ),
            encoding="utf-8",
        )
        (p.with_suffix(".evidence.json")).write_text(
            json.dumps(
                {
                    "question": question,
                    "items": [
                        {
                            "paper_id": "p1",
                            "snippet_id": "P1:S1",
                            "score": 0.8,
                            "section": "abstract",
                            "text": "evidence",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return str(p)

    monkeypatch.setattr("src.apps.research_template.run_research", _fake_research)

    out = run_research_template(
        template_name="lit_review",
        topic="mobile segmentation",
        index_path=str(tmp_path / "idx.json"),
        out_dir=str(tmp_path / "runs" / "sess"),
        session_db=str(tmp_path / "sessions.db"),
        templates_path=str(templates),
    )
    assert out["queries_run"] == 2
    assert len(calls) == 2
    assert Path(out["session_summary_path"]).exists()
    assert Path(out["session_summary_json_path"]).exists()


def test_run_research_template_missing_template_raises(tmp_path) -> None:
    templates = tmp_path / "templates.yaml"
    templates.write_text("templates: {}", encoding="utf-8")
    try:
        run_research_template(
            template_name="x",
            topic="t",
            index_path=str(tmp_path / "idx.json"),
            templates_path=str(templates),
        )
    except ValueError as exc:
        assert "template not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")
