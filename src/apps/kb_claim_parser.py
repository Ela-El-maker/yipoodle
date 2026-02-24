from __future__ import annotations

from dataclasses import dataclass
import re

from src.core.validation import extract_citations


@dataclass
class ParsedClaim:
    claim_text: str
    citations: list[str]


_HEADER_RE = re.compile(r"^##\s+key\s+claims\s*$", re.IGNORECASE)
_NEXT_HEADER_RE = re.compile(r"^##\s+")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")


def parse_key_claims(markdown: str) -> list[ParsedClaim]:
    lines = (markdown or "").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if _HEADER_RE.match(ln.strip()):
            start = i + 1
            break
    if start is None:
        return []

    out: list[ParsedClaim] = []
    for ln in lines[start:]:
        if _NEXT_HEADER_RE.match(ln.strip()):
            break
        if not _BULLET_RE.match(ln):
            continue
        raw = _BULLET_RE.sub("", ln).strip()
        if not raw:
            continue
        cits = extract_citations(raw)
        claim = raw
        for c in cits:
            claim = claim.replace(c, "")
        claim = re.sub(r"\s+", " ", claim).strip(" .;:-\t")
        if not claim:
            continue
        out.append(ParsedClaim(claim_text=claim, citations=sorted(set(cits))))
    return out
