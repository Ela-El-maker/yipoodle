from __future__ import annotations

import re
from datetime import datetime, timezone
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

from src.core.schemas import PaperRecord


def _norm_title_hash(title: str) -> str:
    norm = re.sub(r"\W+", "", title.lower())
    return norm[:32]


def _reconstruct_openalex_abstract(inv_idx: dict | None) -> str:
    if not isinstance(inv_idx, dict) or not inv_idx:
        return ""
    max_pos = max((max(pos_list) for pos_list in inv_idx.values() if pos_list), default=-1)
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inv_idx.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    return " ".join(w for w in words if w).strip()


def _request_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    retries: int = 3,
    backoff_seconds: float = 0.8,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Retry on transient server/network status classes.
            if resp.status_code in {408, 425, 429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"Transient status {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff_seconds * (2**attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed without explicit exception")


def search_arxiv(query: str, max_results: int = 20) -> list[PaperRecord]:
    endpoint = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    papers: list[PaperRecord] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "")
        year = int(published[:4]) if len(published) >= 4 else None
        authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]

        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else None
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        paper_id = f"arxiv:{arxiv_id}" if arxiv_id else f"title:{_norm_title_hash(title)}"
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=[a for a in authors if a],
                year=year,
                venue="arXiv",
                source="arxiv",
                abstract=summary,
                pdf_path=None,
                url=pdf_url or entry_id,
                doi=None,
                arxiv_id=arxiv_id,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_openalex(query: str, max_results: int = 20) -> list[PaperRecord]:
    endpoint = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results,
    }
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    data = r.json().get("results", [])

    papers: list[PaperRecord] = []
    for item in data:
        title = item.get("display_name", "")
        year = item.get("publication_year")
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
        doi = item.get("doi")
        source_id = item.get("id", "")
        paper_id = f"doi:{doi}" if doi else f"openalex:{source_id.rsplit('/', 1)[-1]}"
        best_oa = item.get("best_oa_location") or {}
        primary = item.get("primary_location") or {}
        pdf_url = best_oa.get("pdf_url")
        landing_url = primary.get("landing_page_url") or source_id
        url = pdf_url or landing_url or source_id
        venue = primary.get("source", {}).get("display_name") if primary.get("source") else None
        is_oa = bool(item.get("open_access", {}).get("is_oa") or pdf_url)
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=[a for a in authors if a],
                year=year,
                venue=venue,
                source="openalex",
                abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
                pdf_path=None,
                url=url,
                doi=doi,
                arxiv_id=None,
                openalex_id=source_id,
                citation_count=int(item.get("cited_by_count") or 0),
                is_open_access=is_oa,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_semanticscholar(query: str, max_results: int = 20) -> list[PaperRecord]:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,year,abstract,authors,url,venue,citationCount,externalIds,openAccessPdf,isOpenAccess",
    }
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    data = r.json().get("data", [])

    papers: list[PaperRecord] = []
    for item in data:
        title = item.get("title") or ""
        if not title:
            continue
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        paper_id = f"doi:{doi}" if doi else (f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{_norm_title_hash(title)}")
        pdf_url = (item.get("openAccessPdf") or {}).get("url")
        url = pdf_url or item.get("url") or "https://www.semanticscholar.org/"
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")],
                year=item.get("year"),
                venue=item.get("venue"),
                source="semanticscholar",
                abstract=item.get("abstract") or "",
                pdf_path=None,
                url=url,
                doi=doi,
                arxiv_id=arxiv_id,
                openalex_id=None,
                citation_count=int(item.get("citationCount") or 0),
                is_open_access=bool(item.get("isOpenAccess") or pdf_url),
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_crossref(query: str, max_results: int = 20) -> list[PaperRecord]:
    endpoint = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": max_results,
    }
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    data = (r.json().get("message") or {}).get("items") or []

    papers: list[PaperRecord] = []
    for item in data:
        title_list = item.get("title") or []
        title = str(title_list[0]).strip() if title_list else ""
        if not title:
            continue
        doi = item.get("DOI")
        issued = item.get("issued", {}) or {}
        parts = issued.get("date-parts", []) or []
        year = None
        if parts and parts[0]:
            try:
                year = int(parts[0][0])
            except Exception:
                year = None
        container = item.get("container-title") or []
        venue = str(container[0]).strip() if container else None
        authors = []
        for a in item.get("author") or []:
            given = str(a.get("given") or "").strip()
            family = str(a.get("family") or "").strip()
            name = " ".join(x for x in [given, family] if x).strip()
            if name:
                authors.append(name)
        paper_id = f"doi:{doi}" if doi else f"crossref:{_norm_title_hash(title)}"
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                source="crossref",
                abstract="",
                pdf_path=None,
                url=url,
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=int(item.get("is-referenced-by-count") or 0),
                is_open_access=False,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_dblp(query: str, max_results: int = 20) -> list[PaperRecord]:
    endpoint = "https://dblp.org/search/publ/api"
    params = {
        "q": query,
        "h": max_results,
        "format": "json",
    }
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    result = (r.json().get("result") or {}).get("hits") or {}
    hits = result.get("hit") or []
    if isinstance(hits, dict):
        hits = [hits]

    papers: list[PaperRecord] = []
    for h in hits:
        info = h.get("info") or {}
        title = str(info.get("title") or "").strip()
        if not title:
            continue
        doi = str(info.get("doi") or "").strip() or None
        year_raw = info.get("year")
        try:
            year = int(year_raw) if year_raw is not None else None
        except Exception:
            year = None
        venue = str(info.get("venue") or "").strip() or None
        author_blob = (info.get("authors") or {}).get("author") or []
        if isinstance(author_blob, str):
            authors = [author_blob]
        elif isinstance(author_blob, dict):
            authors = [str(author_blob.get("text") or "").strip()] if author_blob.get("text") else []
        else:
            authors = [str(a).strip() for a in author_blob if str(a).strip()]
        url = str(info.get("url") or "").strip()
        paper_id = f"doi:{doi}" if doi else f"dblp:{_norm_title_hash(title)}"
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                source="dblp",
                abstract="",
                pdf_path=None,
                url=url or f"https://dblp.org/rec/{_norm_title_hash(title)}",
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=False,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_paperswithcode(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    # PWC endpoint has changed over time; prefer configured endpoint, fallback to HF papers API.
    endpoint = endpoint or "https://paperswithcode.com/api/v1/papers/"
    candidates: list[tuple[str, dict]] = []
    if "paperswithcode.com" in endpoint:
        # Fallback to Hugging Face papers endpoint if PWC endpoint is sunset/redirected.
        candidates.append(("https://huggingface.co/api/papers", {"q": query, "limit": max_results}))
        candidates.append(("https://huggingface.co/api/papers", {"query": query, "limit": max_results}))
    else:
        candidates.append((endpoint, {"q": query, "limit": max_results}))
        candidates.append((endpoint, {"query": query, "limit": max_results}))

    last_exc: Exception | None = None
    data = []
    for url, params in candidates:
        try:
            r = _request_get_with_retry(url, params=params, timeout=30)
            try:
                payload = r.json()
            except Exception:
                payload = {}
            if isinstance(payload, list):
                data = payload
            elif isinstance(payload, dict):
                data = payload.get("results") or payload.get("papers") or payload.get("items") or payload.get("data") or []
            else:
                data = []
            if data:
                break
        except Exception as exc:
            last_exc = exc
            continue
    if not data and last_exc:
        raise last_exc

    papers: list[PaperRecord] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        arxiv_id = item.get("arxiv_id") or item.get("arxivId")
        doi = item.get("doi")
        paper_id = (
            f"doi:{doi}"
            if doi
            else (f"arxiv:{arxiv_id}" if arxiv_id else f"paperswithcode:{_norm_title_hash(title)}")
        )
        raw_authors = item.get("authors") or []
        authors: list[str] = []
        if isinstance(raw_authors, list):
            for a in raw_authors:
                if isinstance(a, dict):
                    nm = str(a.get("name") or "").strip()
                    if nm:
                        authors.append(nm)
                else:
                    nm = str(a).strip()
                    if nm:
                        authors.append(nm)
        elif isinstance(raw_authors, str):
            authors = [raw_authors]

        year = item.get("year")
        try:
            year = int(year) if year is not None else None
        except Exception:
            year = None
        venue = str(item.get("venue") or "").strip() or "Papers"
        url = (
            item.get("pdf")
            or item.get("pdf_url")
            or item.get("url")
            or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "https://huggingface.co/papers")
        )
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                source="paperswithcode",
                abstract=str(item.get("abstract") or ""),
                pdf_path=None,
                url=url,
                doi=doi,
                arxiv_id=arxiv_id,
                openalex_id=None,
                citation_count=int(item.get("citation_count") or item.get("citations") or 0),
                is_open_access=bool(".pdf" in str(url)),
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_core(
    query: str,
    max_results: int = 20,
    endpoint: str | None = None,
    auth_header: dict[str, str] | None = None,
) -> list[PaperRecord]:
    endpoint = (endpoint or "https://api.core.ac.uk/v3/search/works").rstrip("/")
    params = {
        "q": query,
        "limit": max_results,
        "offset": 0,
    }
    r = _request_get_with_retry(endpoint, params=params, headers=auth_header, timeout=30)
    try:
        payload = r.json()
    except Exception:
        payload = {}
    data = payload.get("results") or payload.get("data") or payload.get("items") or []
    if isinstance(data, dict):
        data = [data]

    papers: list[PaperRecord] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        doi = item.get("doi")
        year = item.get("yearPublished") or item.get("year")
        try:
            year = int(year) if year is not None else None
        except Exception:
            year = None
        authors_raw = item.get("authors") or []
        authors: list[str] = []
        if isinstance(authors_raw, list):
            for a in authors_raw:
                if isinstance(a, dict):
                    nm = str(a.get("name") or a.get("displayName") or "").strip()
                    if nm:
                        authors.append(nm)
                else:
                    nm = str(a).strip()
                    if nm:
                        authors.append(nm)
        elif isinstance(authors_raw, str):
            authors = [authors_raw]
        venue = str(item.get("publisher") or item.get("journal") or "").strip() or None
        # CORE can expose multiple links; prefer pdf-like links.
        links = item.get("downloadUrl") or item.get("fullText") or item.get("fullTextLink")
        if isinstance(links, list):
            url = ""
            for lk in links:
                s = str(lk)
                if ".pdf" in s.lower():
                    url = s
                    break
            if not url and links:
                url = str(links[0])
        else:
            url = str(links or item.get("sourceFulltextUrls") or item.get("url") or "").strip()
        paper_id = f"doi:{doi}" if doi else f"core:{_norm_title_hash(title)}"
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                source="core",
                abstract=str(item.get("abstract") or ""),
                pdf_path=None,
                url=url or "https://core.ac.uk/",
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=int(item.get("citationCount") or 0),
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_openreview(
    query: str,
    max_results: int = 20,
    endpoint: str | None = None,
) -> list[PaperRecord]:
    # OpenReview API v2 commonly exposes note search-like endpoints.
    # Keep parser resilient because schemas can vary by venue/config.
    endpoint = (endpoint or "https://api2.openreview.net/notes").rstrip("/")
    candidates = [
        {"query": query, "limit": max_results},
        {"term": query, "limit": max_results},
        {"content.title": query, "limit": max_results},
    ]
    data = []
    last_exc: Exception | None = None
    for params in candidates:
        try:
            r = _request_get_with_retry(endpoint, params=params, timeout=30)
            try:
                payload = r.json()
            except Exception:
                payload = {}
            notes = payload.get("notes") if isinstance(payload, dict) else None
            if notes is None and isinstance(payload, dict):
                notes = payload.get("results") or payload.get("items") or payload.get("data")
            if isinstance(notes, list) and notes:
                data = notes
                break
        except Exception as exc:
            last_exc = exc
            continue
    if not data and last_exc:
        raise last_exc

    papers: list[PaperRecord] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or {}
        title = content.get("title")
        if isinstance(title, dict):
            title = title.get("value")
        title = str(title or item.get("title") or "").strip()
        if not title:
            continue
        abstract = content.get("abstract")
        if isinstance(abstract, dict):
            abstract = abstract.get("value")
        abstract = str(abstract or item.get("abstract") or "")
        authors_val = content.get("authors")
        if isinstance(authors_val, dict):
            authors_val = authors_val.get("value")
        authors: list[str] = []
        if isinstance(authors_val, list):
            authors = [str(a).strip() for a in authors_val if str(a).strip()]
        elif isinstance(authors_val, str):
            authors = [authors_val]

        venue_val = content.get("venue") or content.get("venueid")
        if isinstance(venue_val, dict):
            venue_val = venue_val.get("value")
        venue = str(venue_val or "OpenReview").strip()
        year = None
        for key in ("year", "publication_year"):
            y = content.get(key) or item.get(key)
            if isinstance(y, dict):
                y = y.get("value")
            try:
                if y is not None:
                    year = int(y)
                    break
            except Exception:
                pass
        arxiv_id = content.get("arxiv_id") or content.get("arxiv")
        if isinstance(arxiv_id, dict):
            arxiv_id = arxiv_id.get("value")
        arxiv_id = str(arxiv_id).strip() if arxiv_id else None
        forum = item.get("forum") or item.get("id") or item.get("noteId")
        forum = str(forum).strip() if forum else None
        doi = content.get("doi")
        if isinstance(doi, dict):
            doi = doi.get("value")
        doi = str(doi).strip() if doi else None
        paper_id = f"doi:{doi}" if doi else (f"arxiv:{arxiv_id}" if arxiv_id else f"openreview:{forum or _norm_title_hash(title)}")
        pdf_url = ""
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        url = str(item.get("url") or (f"https://openreview.net/forum?id={forum}" if forum else "https://openreview.net"))
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                source="openreview",
                abstract=abstract,
                pdf_path=None,
                url=pdf_url or url,
                doi=doi,
                arxiv_id=arxiv_id,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_github(
    query: str,
    max_results: int = 20,
    endpoint: str | None = None,
    auth_header: dict[str, str] | None = None,
) -> list[PaperRecord]:
    endpoint = endpoint or "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": max_results}
    r = _request_get_with_retry(endpoint, params=params, headers=auth_header, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        items = []

    papers: list[PaperRecord] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        full_name = str(item.get("full_name") or "").strip()
        title = str(item.get("name") or full_name or "").strip()
        if not title:
            continue
        desc = str(item.get("description") or "").strip()
        html_url = str(item.get("html_url") or "").strip()
        created = str(item.get("created_at") or "").strip()
        year = int(created[:4]) if len(created) >= 4 and created[:4].isdigit() else None
        paper_id = f"github:{full_name or _norm_title_hash(title)}"
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title if full_name == "" else full_name,
                authors=[],
                year=year,
                venue="GitHub",
                source="github",
                abstract=desc,
                pdf_path=None,
                url=html_url or "https://github.com/",
                doi=None,
                arxiv_id=None,
                openalex_id=None,
                citation_count=int(item.get("stargazers_count") or 0),
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_zenodo(
    query: str,
    max_results: int = 20,
    endpoint: str | None = None,
) -> list[PaperRecord]:
    endpoint = endpoint or "https://zenodo.org/api/records"
    params = {"q": query, "size": max_results}
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    hits = (payload.get("hits") or {}).get("hits") if isinstance(payload, dict) else []
    if not isinstance(hits, list):
        hits = []

    papers: list[PaperRecord] = []
    for item in hits[:max_results]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        title = str(metadata.get("title") or "").strip()
        if not title:
            continue
        creators = metadata.get("creators") or []
        authors = [str((c or {}).get("name") or "").strip() for c in creators if isinstance(c, dict)]
        authors = [a for a in authors if a]
        doi = str(metadata.get("doi") or item.get("doi") or "").strip() or None
        pub_date = str(metadata.get("publication_date") or "").strip()
        year = int(pub_date[:4]) if len(pub_date) >= 4 and pub_date[:4].isdigit() else None
        files = item.get("files") or []
        file_url = ""
        for f in files:
            if not isinstance(f, dict):
                continue
            links = f.get("links") or {}
            cand = str(links.get("self") or links.get("download") or "").strip()
            if ".pdf" in cand.lower():
                file_url = cand
                break
        links = item.get("links") or {}
        url = file_url or str(links.get("html") or links.get("self") or "").strip()
        paper_id = f"doi:{doi}" if doi else f"zenodo:{item.get('id') or _norm_title_hash(title)}"
        papers.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue="Zenodo",
                source="zenodo",
                abstract=str(metadata.get("description") or ""),
                pdf_path=None,
                url=url or "https://zenodo.org/",
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_opencitations(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    endpoint = endpoint or "https://opencitations.net/index/api/v1/metadata"
    query = query.strip()
    # OpenCitations endpoint is identifier-centric. For free-text queries, return empty.
    if not query or "/" not in query:
        return []
    url = f"{endpoint.rstrip('/')}/{quote(query)}"
    r = _request_get_with_retry(url, timeout=30)
    payload = r.json() if hasattr(r, "json") else []
    data = payload if isinstance(payload, list) else []

    papers: list[PaperRecord] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        author_blob = str(item.get("author") or "").strip()
        authors = [a.strip() for a in author_blob.split(";") if a.strip()]
        year = None
        date_val = str(item.get("pub_date") or "").strip()
        if len(date_val) >= 4 and date_val[:4].isdigit():
            year = int(date_val[:4])
        doi = str(item.get("doi") or query).strip() or None
        url = f"https://doi.org/{doi}" if doi else "https://opencitations.net/"
        papers.append(
            PaperRecord(
                paper_id=f"doi:{doi}" if doi else f"opencitations:{_norm_title_hash(title)}",
                title=title,
                authors=authors,
                year=year,
                venue=None,
                source="opencitations",
                abstract="",
                pdf_path=None,
                url=url,
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=False,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_springer(
    query: str,
    max_results: int = 20,
    endpoint: str | None = None,
    auth_query: dict[str, str] | None = None,
) -> list[PaperRecord]:
    endpoint = endpoint or "https://api.springernature.com/metadata/json"
    params = {"q": query, "p": max_results}
    if auth_query:
        params.update(auth_query)
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    records = payload.get("records") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        records = []

    papers: list[PaperRecord] = []
    for item in records[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        doi = str(item.get("doi") or "").strip() or None
        year = None
        date_val = str(item.get("publicationDate") or "").strip()
        if len(date_val) >= 4 and date_val[:4].isdigit():
            year = int(date_val[:4])
        creators = str(item.get("creators") or "").strip()
        authors = [a.strip() for a in creators.split(";") if a.strip()]
        url = str(item.get("url") or item.get("identifier") or "").strip()
        papers.append(
            PaperRecord(
                paper_id=f"doi:{doi}" if doi else f"springer:{_norm_title_hash(title)}",
                title=title,
                authors=authors,
                year=year,
                venue=str(item.get("publicationName") or "").strip() or None,
                source="springer",
                abstract=str(item.get("abstract") or ""),
                pdf_path=None,
                url=url or (f"https://doi.org/{doi}" if doi else "https://link.springer.com/"),
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=False,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_ieee_xplore(
    query: str,
    max_results: int = 20,
    endpoint: str | None = None,
    auth_query: dict[str, str] | None = None,
) -> list[PaperRecord]:
    endpoint = endpoint or "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    params = {"querytext": query, "max_records": max_results, "format": "json"}
    if auth_query:
        params.update(auth_query)
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    records = payload.get("articles") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        records = []

    papers: list[PaperRecord] = []
    for item in records[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        doi = str(item.get("doi") or "").strip() or None
        year = None
        y = str(item.get("publication_year") or "").strip()
        if y.isdigit():
            year = int(y)
        authors_blob = item.get("authors") or {}
        author_list = (authors_blob.get("authors") if isinstance(authors_blob, dict) else []) or []
        authors = [str((a or {}).get("full_name") or "").strip() for a in author_list if isinstance(a, dict)]
        authors = [a for a in authors if a]
        url = str(item.get("pdf_url") or item.get("html_url") or "").strip()
        papers.append(
            PaperRecord(
                paper_id=f"doi:{doi}" if doi else f"ieee:{_norm_title_hash(title)}",
                title=title,
                authors=authors,
                year=year,
                venue=str(item.get("publication_title") or "").strip() or "IEEE Xplore",
                source="ieee_xplore",
                abstract=str(item.get("abstract") or ""),
                pdf_path=None,
                url=url or "https://ieeexplore.ieee.org/",
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=int(item.get("citing_paper_count") or 0),
                is_open_access=bool(item.get("open_access") or ".pdf" in url.lower()),
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_figshare(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    endpoint = endpoint or "https://api.figshare.com/v2/articles"
    params = {"search_for": query, "page_size": max_results, "order": "desc"}
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    payload = r.json() if hasattr(r, "json") else []
    data = payload if isinstance(payload, list) else payload.get("items", []) if isinstance(payload, dict) else []

    papers: list[PaperRecord] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        doi = str(item.get("doi") or "").strip() or None
        published = str(item.get("published_date") or "").strip()
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        url = str(item.get("url_public_html") or item.get("url") or "").strip()
        papers.append(
            PaperRecord(
                paper_id=f"doi:{doi}" if doi else f"figshare:{item.get('id') or _norm_title_hash(title)}",
                title=title,
                authors=[],
                year=year,
                venue="Figshare",
                source="figshare",
                abstract=str(item.get("description") or ""),
                pdf_path=None,
                url=url or "https://figshare.com/",
                doi=doi,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_openml(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    base = (endpoint or "https://www.openml.org/api/v1/json").rstrip("/")
    url = f"{base}/data/list/data_name/{quote(query)}/limit/{max_results}"
    r = _request_get_with_retry(url, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    data = ((payload.get("data") or {}).get("dataset")) if isinstance(payload, dict) else []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        data = []

    papers: list[PaperRecord] = []
    for item in data[:max_results]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        did = str(item.get("did") or "").strip()
        url = f"https://www.openml.org/d/{did}" if did else "https://www.openml.org/"
        papers.append(
            PaperRecord(
                paper_id=f"openml:{did or _norm_title_hash(name)}",
                title=f"OpenML dataset: {name}",
                authors=[],
                year=None,
                venue="OpenML",
                source="openml",
                abstract=str(item.get("description") or ""),
                pdf_path=None,
                url=url,
                doi=None,
                arxiv_id=None,
                openalex_id=None,
                citation_count=int(item.get("NumberOfDownloads") or 0),
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_gdelt(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    endpoint = endpoint or "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": query, "mode": "ArtList", "maxrecords": max_results, "format": "json"}
    r = _request_get_with_retry(endpoint, params=params, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    articles = payload.get("articles") if isinstance(payload, dict) else []
    if not isinstance(articles, list):
        articles = []

    papers: list[PaperRecord] = []
    for item in articles[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        url = str(item.get("url") or "").strip()
        date_val = str(item.get("seendate") or item.get("date") or "").strip()
        year = int(date_val[:4]) if len(date_val) >= 4 and date_val[:4].isdigit() else None
        papers.append(
            PaperRecord(
                paper_id=f"gdelt:{_norm_title_hash(title + url)}",
                title=title,
                authors=[],
                year=year,
                venue=str(item.get("domain") or "GDELT"),
                source="gdelt",
                abstract=str(item.get("socialimage") or ""),
                pdf_path=None,
                url=url or "https://www.gdeltproject.org/",
                doi=None,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_wikidata(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    endpoint = endpoint or "https://query.wikidata.org/sparql"
    safe_q = query.replace('"', '\\"')
    sparql = (
        "SELECT ?item ?itemLabel ?desc WHERE { "
        '?item wdt:P31 wd:Q13442814 . '
        f'?item rdfs:label ?itemLabel FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{safe_q}"))) . '
        "OPTIONAL { ?item schema:description ?desc FILTER(LANG(?desc)='en') } "
        "FILTER(LANG(?itemLabel)='en') } LIMIT "
        f"{max_results}"
    )
    headers = {"Accept": "application/sparql-results+json"}
    r = _request_get_with_retry(endpoint, params={"query": sparql, "format": "json"}, headers=headers, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    binds = (((payload.get("results") or {}).get("bindings")) if isinstance(payload, dict) else []) or []
    if not isinstance(binds, list):
        binds = []

    papers: list[PaperRecord] = []
    for row in binds[:max_results]:
        if not isinstance(row, dict):
            continue
        title = str(((row.get("itemLabel") or {}).get("value") or "")).strip()
        if not title:
            continue
        url = str(((row.get("item") or {}).get("value") or "")).strip()
        desc = str(((row.get("desc") or {}).get("value") or "")).strip()
        papers.append(
            PaperRecord(
                paper_id=f"wikidata:{url.rsplit('/', 1)[-1] or _norm_title_hash(title)}",
                title=title,
                authors=[],
                year=None,
                venue="Wikidata",
                source="wikidata",
                abstract=desc,
                pdf_path=None,
                url=url or "https://www.wikidata.org/",
                doi=None,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers


def search_orcid(query: str, max_results: int = 20, endpoint: str | None = None) -> list[PaperRecord]:
    endpoint = endpoint or "https://pub.orcid.org/v3.0/expanded-search/"
    headers = {"Accept": "application/json"}
    params = {"q": query, "rows": max_results}
    r = _request_get_with_retry(endpoint, params=params, headers=headers, timeout=30)
    payload = r.json() if hasattr(r, "json") else {}
    expanded = payload.get("expanded-result") if isinstance(payload, dict) else []
    if not isinstance(expanded, list):
        expanded = []

    papers: list[PaperRecord] = []
    for item in expanded[:max_results]:
        if not isinstance(item, dict):
            continue
        orcid_id = str(item.get("orcid-id") or "").strip()
        family = str(item.get("family-names") or "").strip()
        given = str(item.get("given-names") or "").strip()
        name = " ".join(x for x in [given, family] if x).strip() or orcid_id
        if not name:
            continue
        papers.append(
            PaperRecord(
                paper_id=f"orcid:{orcid_id or _norm_title_hash(name)}",
                title=f"ORCID profile: {name}",
                authors=[name] if name else [],
                year=None,
                venue="ORCID",
                source="orcid",
                abstract="Author profile result",
                pdf_path=None,
                url=f"https://orcid.org/{orcid_id}" if orcid_id else "https://orcid.org/",
                doi=None,
                arxiv_id=None,
                openalex_id=None,
                citation_count=0,
                is_open_access=True,
                sync_timestamp=datetime.now(timezone.utc),
            )
        )
    return papers
