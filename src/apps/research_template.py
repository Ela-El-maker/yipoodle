from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

import yaml

from src.apps.research_copilot import run_research
from src.apps.session_store import (
    DEFAULT_SESSION_DB,
    create_session,
    get_session_detail,
    record_query,
    session_summary,
    render_session_markdown,
)
from src.core.schemas import EvidencePack, ResearchReport


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "topic"


def load_templates(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_report_meta(report_path: Path) -> dict[str, Any]:
    report_json_path = report_path.with_suffix(".json")
    evidence_path = report_path.with_suffix(".evidence.json")
    if not report_json_path.exists() or not evidence_path.exists():
        return {
            "key_claims": [],
            "gaps": [],
            "citations": [],
            "synthesis_preview": "",
            "evidence_count": 0,
        }
    rep = ResearchReport(**json.loads(report_json_path.read_text(encoding="utf-8")))
    ev = EvidencePack(**json.loads(evidence_path.read_text(encoding="utf-8")))
    citations: list[str] = []
    for it in ev.items:
        if isinstance(it.snippet_id, str) and it.snippet_id:
            citations.append(it.snippet_id)
    return {
        "key_claims": list(rep.key_claims),
        "gaps": list(rep.gaps),
        "citations": sorted(set(citations)),
        "synthesis_preview": rep.synthesis[:500],
        "evidence_count": len(ev.items),
    }


def run_research_template(
    *,
    template_name: str,
    topic: str,
    index_path: str,
    out_dir: str | None = None,
    session_db: str = DEFAULT_SESSION_DB,
    templates_path: str = "config/templates.yaml",
    sources_config_path: str | None = None,
    top_k: int = 8,
    min_items: int = 2,
    min_score: float = 0.5,
    retrieval_mode: str = "hybrid",
    alpha: float = 0.6,
    max_per_paper: int = 2,
    quality_prior_weight: float = 0.15,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    cfg = load_templates(templates_path)
    templates = cfg.get("templates", {}) if isinstance(cfg, dict) else {}
    if not isinstance(templates, dict) or template_name not in templates:
        raise ValueError(f"template not found: {template_name}")
    template = templates.get(template_name) or {}
    questions = template.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"template has no questions: {template_name}")

    session_name = f"{template_name}:{_slug(topic)}:{_utc_ts()}"
    session_id = create_session(session_db, session_name, description=str(template.get("description", "")))

    root = Path(out_dir or (Path("runs/sessions") / str(session_id)))
    root.mkdir(parents=True, exist_ok=True)

    query_outputs: list[dict[str, Any]] = []
    for idx, raw_q in enumerate(questions, start=1):
        question = str(raw_q).format(topic=topic).strip()
        report_path = root / f"q{idx:02d}.md"
        report_out = run_research(
            index_path=index_path,
            question=question,
            top_k=top_k,
            out_path=str(report_path),
            min_items=min_items,
            min_score=min_score,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            max_per_paper=max_per_paper,
            quality_prior_weight=quality_prior_weight,
            embedding_model=embedding_model,
            sources_config_path=sources_config_path,
        )
        meta = _extract_report_meta(Path(report_out))
        query_id = record_query(
            session_db,
            session_id,
            question=question,
            report_path=str(report_out),
            key_claims=meta["key_claims"],
            gaps=meta["gaps"],
            citations=meta["citations"],
            synthesis_preview=meta["synthesis_preview"],
            evidence_count=int(meta["evidence_count"]),
            aggregation_enabled=True,
        )
        query_outputs.append(
            {
                "query_id": query_id,
                "question": question,
                "report_path": str(report_out),
            }
        )

    sess_summary = session_summary(session_db, session_id)
    detail = get_session_detail(session_db, session_id)
    summary_md = render_session_markdown(detail, sess_summary)
    summary_path = root / "session_summary.md"
    summary_json_path = root / "session_summary.json"
    summary_path.write_text(summary_md, encoding="utf-8")
    summary_json_path.write_text(json.dumps(sess_summary.__dict__, indent=2), encoding="utf-8")

    return {
        "template": template_name,
        "topic": topic,
        "session_id": session_id,
        "session_name": session_name,
        "queries_run": len(query_outputs),
        "outputs": query_outputs,
        "session_summary_path": str(summary_path),
        "session_summary_json_path": str(summary_json_path),
        "session_db": session_db,
    }
