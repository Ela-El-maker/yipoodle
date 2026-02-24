from __future__ import annotations

from pathlib import Path
import re
import sqlite3

import requests

from src.core.schemas import PaperRecord


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                year INTEGER,
                venue TEXT,
                source TEXT NOT NULL,
                abstract TEXT,
                pdf_path TEXT,
                url TEXT NOT NULL,
                doi TEXT,
                arxiv_id TEXT,
                openalex_id TEXT,
                citation_count INTEGER DEFAULT 0,
                is_open_access INTEGER DEFAULT 0,
                sync_timestamp TEXT NOT NULL,
                dedupe_key TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_dedupe_key ON papers(dedupe_key)")
        # Lightweight forward-compatible migration for existing DBs.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
        if "openalex_id" not in existing_cols:
            conn.execute("ALTER TABLE papers ADD COLUMN openalex_id TEXT")
        if "citation_count" not in existing_cols:
            conn.execute("ALTER TABLE papers ADD COLUMN citation_count INTEGER DEFAULT 0")
        if "is_open_access" not in existing_cols:
            conn.execute("ALTER TABLE papers ADD COLUMN is_open_access INTEGER DEFAULT 0")


def _normalized_title_key(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())[:64]


def dedupe_key_for_paper(p: PaperRecord) -> str:
    if p.doi:
        return f"doi:{p.doi.lower()}"
    if p.arxiv_id:
        return f"arxiv:{p.arxiv_id.lower()}"
    return f"title:{_normalized_title_key(p.title)}"


def upsert_papers(db_path: str, papers: list[PaperRecord]) -> int:
    added = 0
    with sqlite3.connect(db_path) as conn:
        for p in papers:
            key = dedupe_key_for_paper(p)
            existing = conn.execute("SELECT paper_id FROM papers WHERE dedupe_key = ?", (key,)).fetchone()
            if existing:
                continue
            conn.execute(
                """
                INSERT INTO papers(
                    paper_id, title, authors, year, venue, source, abstract, pdf_path, url, doi, arxiv_id,
                    openalex_id, citation_count, is_open_access, sync_timestamp, dedupe_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    p.paper_id,
                    p.title,
                    "|".join(p.authors),
                    p.year,
                    p.venue,
                    p.source,
                    p.abstract,
                    p.pdf_path,
                    str(p.url),
                    p.doi,
                    p.arxiv_id,
                    p.openalex_id,
                    int(p.citation_count),
                    int(p.is_open_access),
                    p.sync_timestamp.isoformat(),
                    key,
                ),
            )
            added += 1
        conn.commit()
    return added


def fetch_papers(db_path: str) -> list[PaperRecord]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT paper_id, title, authors, year, venue, source, abstract, pdf_path, url, doi, arxiv_id, openalex_id, citation_count, is_open_access, sync_timestamp FROM papers"
        ).fetchall()

    out = []
    for r in rows:
        out.append(
            PaperRecord(
                paper_id=r[0],
                title=r[1],
                authors=r[2].split("|") if r[2] else [],
                year=r[3],
                venue=r[4],
                source=r[5],
                abstract=r[6] or "",
                pdf_path=r[7],
                url=r[8],
                doi=r[9],
                arxiv_id=r[10],
                openalex_id=r[11],
                citation_count=int(r[12] or 0),
                is_open_access=bool(r[13]),
                sync_timestamp=r[14],
            )
        )
    return out


def download_pdf_with_status(url: str, dest: str) -> str:
    if not url:
        return "missing_pdf_url"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code in {401, 403}:
            return "blocked_or_paywalled"
        if r.status_code != 200:
            return "download_http_error"
        if "pdf" not in r.headers.get("content-type", "").lower():
            return "non_pdf_content_type"
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(r.content)
        return "downloaded"
    except Exception:
        return "download_http_error"


def download_pdf(url: str, dest: str) -> bool:
    return download_pdf_with_status(url, dest) == "downloaded"
