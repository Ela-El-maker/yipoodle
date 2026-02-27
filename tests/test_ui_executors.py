from __future__ import annotations

from pathlib import Path

from src.ui.executors import UIExecutors
from src.ui.settings import UISettings


def test_executors_ask_mode(tmp_path: Path) -> None:
    settings = UISettings.model_validate(
        {
            "run_db_path": str(tmp_path / "runs.db"),
            "artifacts_roots": [str(tmp_path)],
        }
    )
    ex = UIExecutors(settings)

    req = {
        "mode": "ask",
        "question": "23 + 34 = ?",
        "output_path": str(tmp_path / "ask.md"),
        "options": {},
    }
    out = ex.execute(
        run_id="r1",
        request=req,
        emit_event=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
    )

    assert Path(str(out.output_path)).exists()
    assert out.details["mode"] == "ask"
    assert out.details["artifacts"]["report_path"].endswith("ask.md")


def test_executors_query_writes_router_sidecar(tmp_path: Path) -> None:
    settings = UISettings.model_validate(
        {
            "run_db_path": str(tmp_path / "runs.db"),
            "artifacts_roots": [str(tmp_path), "runs/query", "runs/research_reports", "runs/monitor", "runs/notes"],
        }
    )
    ex = UIExecutors(settings)

    req = {
        "mode": "query",
        "question": "What is an algorithm?",
        "output_path": str(tmp_path / "query.md"),
        "options": {"mode": "ask"},
        "sources_config": "config/sources.yaml",
        "automation_config": "config/automation.yaml",
        "index": "data/indexes/bm25_index.json",
    }
    out = ex.execute(
        run_id="r2",
        request=req,
        emit_event=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
    )

    assert Path(str(out.output_path)).exists()
    router_sidecar = Path(str(out.output_path)).with_suffix(".router.json")
    assert router_sidecar.exists()
    assert out.details["artifacts"]["router_sidecar_path"] == str(router_sidecar)
