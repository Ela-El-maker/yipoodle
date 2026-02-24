from __future__ import annotations

import json
from pathlib import Path

from src.core.schemas import EvidencePack, ResearchReport


def save_report_json(report: ResearchReport, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return str(p)


def load_evidence_pack(path: str) -> EvidencePack:
    return EvidencePack(**json.loads(Path(path).read_text(encoding="utf-8")))
