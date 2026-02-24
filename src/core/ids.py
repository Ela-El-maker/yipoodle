from __future__ import annotations

import re


def normalize_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")


def snippet_id(paper_id: str, n: int) -> str:
    return f"P{normalize_id(paper_id)}:S{n}"
