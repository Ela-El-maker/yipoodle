from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    source: str
    abstract: str = ""
    pdf_path: str | None = None
    url: HttpUrl | str
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    citation_count: int = 0
    is_open_access: bool = False
    sync_timestamp: datetime


class SnippetRecord(BaseModel):
    snippet_id: str
    paper_id: str
    section: str
    text: str
    page_hint: int | None = None
    token_count: int
    paper_year: int | None = None
    paper_venue: str | None = None
    citation_count: int = 0
    extraction_quality_score: float | None = None
    extraction_quality_band: Literal["good", "ok", "poor"] | None = None
    extraction_source: Literal["native", "ocr"] | None = None


class EvidenceItem(BaseModel):
    paper_id: str
    snippet_id: str
    score: float
    section: str
    text: str
    paper_year: int | None = None
    paper_venue: str | None = None
    citation_count: int = 0
    extraction_quality_score: float | None = None
    extraction_quality_band: Literal["good", "ok", "poor"] | None = None
    extraction_source: Literal["native", "ocr"] | None = None


class EvidencePack(BaseModel):
    question: str
    items: list[EvidenceItem]


class ShortlistItem(BaseModel):
    paper_id: str
    title: str
    reason: str


class ExperimentProposal(BaseModel):
    proposal: str
    citations: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    question: str
    shortlist: list[ShortlistItem] = Field(default_factory=list)
    synthesis: str
    key_claims: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    experiments: list[ExperimentProposal] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    retrieval_diagnostics: dict[str, Any] | None = None


class DocType(str):
    READ_ME = "readme"
    ARCH = "arch"
    API = "api"


class DocWriteRequest(BaseModel):
    doc_type: Literal["readme", "arch", "api"]
    facts: dict


class ReleaseNotesRequest(BaseModel):
    from_ref: str
    to_ref: str
    commits: list[str]
