from __future__ import annotations

from pathlib import Path
import json
import re
import requests

from src.apps.paper_ingest import download_pdf_with_status, init_db, upsert_papers
from src.apps.paper_search import (
    search_arxiv,
    search_core,
    search_crossref,
    search_dblp,
    search_figshare,
    search_gdelt,
    search_github,
    search_ieee_xplore,
    search_openml,
    search_opencitations,
    search_openalex,
    search_openreview,
    search_orcid,
    search_paperswithcode,
    search_semanticscholar,
    search_springer,
    search_wikidata,
    search_zenodo,
)
from src.apps.sources_config import (
    source_auth_query_param,
    source_auth_header,
    load_sources_config,
    source_endpoint,
    source_enabled,
    source_max_results,
    source_required_param,
    unsupported_enabled_sources,
)


_DOI_URL_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


def _has_pdf_candidate(url: str, source: str) -> bool:
    lowered = (url or "").lower()
    if source == "arxiv":
        return True
    return ".pdf" in lowered or lowered.endswith("/pdf")


def _normalize_doi(raw: str | None) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip()
    if not val:
        return None
    val = _DOI_URL_RE.sub("", val).strip()
    if val.lower().startswith("doi:"):
        val = val[4:].strip()
    val = val.strip().strip("/")
    return val.lower() if val else None


def _extract_doi_from_url(url: str | None) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    m = re.search(r"(?:https?://)?(?:dx\.)?doi\.org/([^?#]+)", raw, flags=re.IGNORECASE)
    if not m:
        return None
    return _normalize_doi(m.group(1))


def _unpaywall_pdf_url(doi: str, endpoint: str, email: str, timeout: int = 20) -> tuple[str | None, bool]:
    clean = str(doi).strip()
    if not clean:
        return None, False
    url = f"{endpoint.rstrip('/')}/{clean}"
    try:
        r = requests.get(url, params={"email": email}, timeout=timeout)
        if r.status_code >= 400:
            return None, False
        payload = r.json() if r.text else {}
        best = payload.get("best_oa_location") or {}
        locs = payload.get("oa_locations") or []
        pdf = best.get("url_for_pdf") or best.get("url")
        if not pdf:
            for loc in locs:
                pdf = (loc or {}).get("url_for_pdf") or (loc or {}).get("url")
                if pdf:
                    break
        is_oa = bool(payload.get("is_oa") or pdf)
        return (str(pdf) if pdf else None), is_oa
    except Exception:
        return None, False


def _load_unpaywall_cache(path: str | None) -> dict[str, dict[str, object]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for k, v in payload.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def _save_unpaywall_cache(path: str | None, payload: dict[str, dict[str, object]]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _evaluate_source_quality(stats: dict[str, object], cfg: dict[str, object]) -> tuple[bool, list[str], dict[str, float]]:
    limits = (cfg.get("limits") or {}) if isinstance(cfg, dict) else {}
    sq = (limits.get("source_quality") or {}) if isinstance(limits, dict) else {}
    if not isinstance(sq, dict):
        sq = {}

    fetched = int(stats.get("fetched", 0) or 0)
    downloads_attempted = int(stats.get("downloads_attempted", 0) or 0)
    downloaded = int(stats.get("downloaded", 0) or 0)
    source_errors = int(stats.get("source_errors", 0) or 0)
    download_http_error = int(stats.get("download_http_error", 0) or 0)
    non_pdf_content_type = int(stats.get("non_pdf_content_type", 0) or 0)
    blocked_or_paywalled = int(stats.get("blocked_or_paywalled", 0) or 0)
    unpaywall_lookups = int(stats.get("unpaywall_lookups", 0) or 0)
    unpaywall_lookup_errors = int(stats.get("unpaywall_lookup_errors", 0) or 0)

    metrics = {
        "source_error_rate": _safe_rate(source_errors, max(fetched, 1)),
        "download_success_rate": _safe_rate(downloaded, max(downloads_attempted, 1)),
        "download_http_error_rate": _safe_rate(download_http_error, max(downloads_attempted, 1)),
        "non_pdf_content_rate": _safe_rate(non_pdf_content_type, max(downloads_attempted, 1)),
        "blocked_or_paywalled_rate": _safe_rate(blocked_or_paywalled, max(downloads_attempted, 1)),
        "unpaywall_lookup_error_rate": _safe_rate(unpaywall_lookup_errors, max(unpaywall_lookups, 1)),
    }

    reasons: list[str] = []

    max_source_error_rate = sq.get("max_source_error_rate")
    if max_source_error_rate is not None and metrics["source_error_rate"] > float(max_source_error_rate):
        reasons.append(
            f"source_error_rate_above_threshold:{metrics['source_error_rate']:.3f}>{float(max_source_error_rate):.3f}"
        )

    min_download_success_rate = sq.get("min_download_success_rate")
    if (
        min_download_success_rate is not None
        and downloads_attempted > 0
        and metrics["download_success_rate"] < float(min_download_success_rate)
    ):
        reasons.append(
            f"download_success_rate_below_threshold:{metrics['download_success_rate']:.3f}<{float(min_download_success_rate):.3f}"
        )

    max_download_http_error_rate = sq.get("max_download_http_error_rate")
    if (
        max_download_http_error_rate is not None
        and downloads_attempted > 0
        and metrics["download_http_error_rate"] > float(max_download_http_error_rate)
    ):
        reasons.append(
            f"download_http_error_rate_above_threshold:{metrics['download_http_error_rate']:.3f}>{float(max_download_http_error_rate):.3f}"
        )

    max_non_pdf_content_rate = sq.get("max_non_pdf_content_rate")
    if (
        max_non_pdf_content_rate is not None
        and downloads_attempted > 0
        and metrics["non_pdf_content_rate"] > float(max_non_pdf_content_rate)
    ):
        reasons.append(
            f"non_pdf_content_rate_above_threshold:{metrics['non_pdf_content_rate']:.3f}>{float(max_non_pdf_content_rate):.3f}"
        )

    max_blocked_or_paywalled_rate = sq.get("max_blocked_or_paywalled_rate")
    if (
        max_blocked_or_paywalled_rate is not None
        and downloads_attempted > 0
        and metrics["blocked_or_paywalled_rate"] > float(max_blocked_or_paywalled_rate)
    ):
        reasons.append(
            f"blocked_or_paywalled_rate_above_threshold:{metrics['blocked_or_paywalled_rate']:.3f}>{float(max_blocked_or_paywalled_rate):.3f}"
        )

    max_unpaywall_lookup_error_rate = sq.get("max_unpaywall_lookup_error_rate")
    if (
        max_unpaywall_lookup_error_rate is not None
        and unpaywall_lookups > 0
        and metrics["unpaywall_lookup_error_rate"] > float(max_unpaywall_lookup_error_rate)
    ):
        reasons.append(
            f"unpaywall_lookup_error_rate_above_threshold:{metrics['unpaywall_lookup_error_rate']:.3f}>{float(max_unpaywall_lookup_error_rate):.3f}"
        )

    return len(reasons) == 0, reasons, metrics


def sync_papers(
    query: str,
    max_results: int,
    db_path: str,
    papers_dir: str,
    with_semantic_scholar: bool = False,
    prefer_arxiv: bool = False,
    require_pdf: bool = False,
    sources_config_path: str | None = None,
) -> dict[str, object]:
    sources_cfg = load_sources_config(sources_config_path)
    cfg_sources = (sources_cfg.get("sources") or {}) if isinstance(sources_cfg, dict) else {}
    max_total_results = int((sources_cfg.get("limits", {}) or {}).get("max_total_results", 0) or 0)
    max_pdf_downloads = int((sources_cfg.get("limits", {}) or {}).get("max_pdf_downloads", 0) or 0)
    unpaywall_enabled = bool(source_enabled(sources_cfg, "unpaywall"))
    unpaywall_endpoint = source_endpoint(sources_cfg, "unpaywall", default="https://api.unpaywall.org/v2")
    unpaywall_email = source_required_param(sources_cfg, "unpaywall", "email")

    init_db(db_path)
    errors = 0
    arxiv: list = []
    openalex: list = []
    semscholar: list = []
    crossref: list = []
    dblp: list = []
    paperswithcode: list = []
    core: list = []
    openreview: list = []
    github: list = []
    zenodo: list = []
    opencitations: list = []
    springer: list = []
    ieee_xplore: list = []
    figshare: list = []
    openml: list = []
    gdelt: list = []
    wikidata: list = []
    orcid: list = []

    if source_enabled(sources_cfg, "arxiv"):
        try:
            arxiv = search_arxiv(query, max_results=source_max_results(sources_cfg, "arxiv", max_results))
        except Exception:
            arxiv = []
            errors += 1
    if source_enabled(sources_cfg, "openalex"):
        try:
            openalex = search_openalex(query, max_results=source_max_results(sources_cfg, "openalex", max_results))
        except Exception:
            openalex = []
            errors += 1
    if with_semantic_scholar and source_enabled(sources_cfg, "semanticscholar"):
        try:
            semscholar = search_semanticscholar(
                query, max_results=source_max_results(sources_cfg, "semanticscholar", max_results)
            )
        except Exception:
            semscholar = []
            errors += 1
    if "crossref" in cfg_sources and source_enabled(sources_cfg, "crossref"):
        try:
            crossref = search_crossref(query, max_results=source_max_results(sources_cfg, "crossref", max_results))
        except Exception:
            crossref = []
            errors += 1
    if "dblp" in cfg_sources and source_enabled(sources_cfg, "dblp"):
        try:
            dblp = search_dblp(query, max_results=source_max_results(sources_cfg, "dblp", max_results))
        except Exception:
            dblp = []
            errors += 1
    if "paperswithcode" in cfg_sources and source_enabled(sources_cfg, "paperswithcode"):
        try:
            paperswithcode = search_paperswithcode(
                query,
                max_results=source_max_results(sources_cfg, "paperswithcode", max_results),
                endpoint=source_endpoint(sources_cfg, "paperswithcode", default=None),
            )
        except Exception:
            paperswithcode = []
            errors += 1
    if "core" in cfg_sources and source_enabled(sources_cfg, "core"):
        try:
            core = search_core(
                query,
                max_results=source_max_results(sources_cfg, "core", max_results),
                endpoint=source_endpoint(sources_cfg, "core", default=None),
                auth_header=source_auth_header(sources_cfg, "core"),
            )
        except Exception:
            core = []
            errors += 1
    if "openreview" in cfg_sources and source_enabled(sources_cfg, "openreview"):
        try:
            openreview = search_openreview(
                query,
                max_results=source_max_results(sources_cfg, "openreview", max_results),
                endpoint=source_endpoint(sources_cfg, "openreview", default=None),
            )
        except Exception:
            openreview = []
            errors += 1
    if "github" in cfg_sources and source_enabled(sources_cfg, "github"):
        try:
            github = search_github(
                query,
                max_results=source_max_results(sources_cfg, "github", max_results),
                endpoint=source_endpoint(sources_cfg, "github", default=None),
                auth_header=source_auth_header(sources_cfg, "github"),
            )
        except Exception:
            github = []
            errors += 1
    if "zenodo" in cfg_sources and source_enabled(sources_cfg, "zenodo"):
        try:
            zenodo = search_zenodo(
                query,
                max_results=source_max_results(sources_cfg, "zenodo", max_results),
                endpoint=source_endpoint(sources_cfg, "zenodo", default=None),
            )
        except Exception:
            zenodo = []
            errors += 1
    if "opencitations" in cfg_sources and source_enabled(sources_cfg, "opencitations"):
        try:
            opencitations = search_opencitations(
                query,
                max_results=source_max_results(sources_cfg, "opencitations", max_results),
                endpoint=source_endpoint(sources_cfg, "opencitations", default=None),
            )
        except Exception:
            opencitations = []
            errors += 1
    if "springer" in cfg_sources and source_enabled(sources_cfg, "springer"):
        try:
            springer = search_springer(
                query,
                max_results=source_max_results(sources_cfg, "springer", max_results),
                endpoint=source_endpoint(sources_cfg, "springer", default=None),
                auth_query=source_auth_query_param(sources_cfg, "springer"),
            )
        except Exception:
            springer = []
            errors += 1
    if "ieee_xplore" in cfg_sources and source_enabled(sources_cfg, "ieee_xplore"):
        try:
            ieee_xplore = search_ieee_xplore(
                query,
                max_results=source_max_results(sources_cfg, "ieee_xplore", max_results),
                endpoint=source_endpoint(sources_cfg, "ieee_xplore", default=None),
                auth_query=source_auth_query_param(sources_cfg, "ieee_xplore"),
            )
        except Exception:
            ieee_xplore = []
            errors += 1
    if "figshare" in cfg_sources and source_enabled(sources_cfg, "figshare"):
        try:
            figshare = search_figshare(
                query,
                max_results=source_max_results(sources_cfg, "figshare", max_results),
                endpoint=source_endpoint(sources_cfg, "figshare", default=None),
            )
        except Exception:
            figshare = []
            errors += 1
    if "openml" in cfg_sources and source_enabled(sources_cfg, "openml"):
        try:
            openml = search_openml(
                query,
                max_results=source_max_results(sources_cfg, "openml", max_results),
                endpoint=source_endpoint(sources_cfg, "openml", default=None),
            )
        except Exception:
            openml = []
            errors += 1
    if "gdelt" in cfg_sources and source_enabled(sources_cfg, "gdelt"):
        try:
            gdelt = search_gdelt(
                query,
                max_results=source_max_results(sources_cfg, "gdelt", max_results),
                endpoint=source_endpoint(sources_cfg, "gdelt", default=None),
            )
        except Exception:
            gdelt = []
            errors += 1
    if "wikidata" in cfg_sources and source_enabled(sources_cfg, "wikidata"):
        try:
            wikidata = search_wikidata(
                query,
                max_results=source_max_results(sources_cfg, "wikidata", max_results),
                endpoint=source_endpoint(sources_cfg, "wikidata", default=None),
            )
        except Exception:
            wikidata = []
            errors += 1
    if "orcid" in cfg_sources and source_enabled(sources_cfg, "orcid"):
        try:
            orcid = search_orcid(
                query,
                max_results=source_max_results(sources_cfg, "orcid", max_results),
                endpoint=source_endpoint(sources_cfg, "orcid", default=None),
            )
        except Exception:
            orcid = []
            errors += 1

    merged = (
        arxiv
        + openalex
        + semscholar
        + crossref
        + dblp
        + paperswithcode
        + core
        + openreview
        + github
        + zenodo
        + opencitations
        + springer
        + ieee_xplore
        + figshare
        + openml
        + gdelt
        + wikidata
        + orcid
    )
    if prefer_arxiv:
        merged = sorted(merged, key=lambda p: 0 if p.source == "arxiv" else 1)
    if max_total_results > 0:
        merged = merged[:max_total_results]
    if require_pdf:
        merged = [p for p in merged if _has_pdf_candidate(str(p.url), p.source)]

    normalized_merged = []
    for p in merged:
        norm_doi = _normalize_doi(p.doi) or _extract_doi_from_url(str(p.url or ""))
        if norm_doi:
            update = {"doi": norm_doi}
            if str(p.paper_id).lower().startswith("doi:"):
                update["paper_id"] = f"doi:{norm_doi}"
            p = p.model_copy(update=update)
        normalized_merged.append(p)
    merged = normalized_merged

    # Optional DOI->OA enrichment via Unpaywall for papers lacking direct PDF-like URLs.
    unpaywall_enriched = 0
    unpaywall_lookups = 0
    unpaywall_lookup_errors = 0
    unpaywall_cache_hits = 0
    unpaywall_cache_misses = 0
    unpaywall_cache_path = str((sources_cfg.get("cache", {}) or {}).get("unpaywall_cache_path", "runs/cache/unpaywall.json"))
    unpaywall_cache = _load_unpaywall_cache(unpaywall_cache_path)
    if unpaywall_enabled and unpaywall_endpoint and unpaywall_email:
        enriched = []
        for p in merged:
            current_url = str(p.url or "")
            norm_doi = _normalize_doi(p.doi) or _extract_doi_from_url(current_url)
            if norm_doi and not _has_pdf_candidate(current_url, p.source):
                unpaywall_lookups += 1
                cached = unpaywall_cache.get(norm_doi)
                if isinstance(cached, dict):
                    unpaywall_cache_hits += 1
                    pdf_url = str(cached.get("pdf_url") or "") or None
                    is_oa = bool(cached.get("is_oa", False))
                else:
                    unpaywall_cache_misses += 1
                    pdf_url, is_oa = _unpaywall_pdf_url(norm_doi, unpaywall_endpoint, unpaywall_email)
                    unpaywall_cache[norm_doi] = {"pdf_url": pdf_url, "is_oa": bool(is_oa)}
                if pdf_url:
                    p = p.model_copy(update={"url": pdf_url, "is_open_access": bool(p.is_open_access or is_oa)})
                    unpaywall_enriched += 1
                else:
                    unpaywall_lookup_errors += 1
            enriched.append(p)
        merged = enriched
        _save_unpaywall_cache(unpaywall_cache_path, unpaywall_cache)

    added = upsert_papers(db_path, merged)

    downloaded = 0
    missing_pdf_url = 0
    download_http_error = 0
    non_pdf_content_type = 0
    blocked_or_paywalled = 0
    downloads_attempted = 0
    require_pdf_filtered = (
        len(arxiv)
        + len(openalex)
        + len(semscholar)
        + len(crossref)
        + len(dblp)
        + len(paperswithcode)
        + len(core)
        + len(openreview)
        + len(github)
        + len(zenodo)
        + len(opencitations)
        + len(springer)
        + len(ieee_xplore)
        + len(figshare)
        + len(openml)
        + len(gdelt)
        + len(wikidata)
        + len(orcid)
        - len(merged)
    )
    unsupported = unsupported_enabled_sources(sources_cfg)

    for i, p in enumerate(merged):
        if max_pdf_downloads > 0 and i >= max_pdf_downloads:
            break
        url = str(p.url)
        if not url:
            missing_pdf_url += 1
            continue
        dest = Path(papers_dir) / f"{p.paper_id.replace(':', '_').replace('/', '_')}.pdf"
        if dest.exists():
            continue
        downloads_attempted += 1
        status = download_pdf_with_status(url, str(dest))
        if status == "downloaded":
            downloaded += 1
        elif status == "missing_pdf_url":
            missing_pdf_url += 1
        elif status == "download_http_error":
            download_http_error += 1
        elif status == "non_pdf_content_type":
            non_pdf_content_type += 1
        elif status == "blocked_or_paywalled":
            blocked_or_paywalled += 1

    stats = {
        "fetched": len(merged),
        "added": added,
        "downloads_attempted": downloads_attempted,
        "downloaded": downloaded,
        "from_arxiv": len(arxiv),
        "from_openalex": len(openalex),
        "from_semanticscholar": len(semscholar),
        "from_crossref": len(crossref),
        "from_dblp": len(dblp),
        "from_paperswithcode": len(paperswithcode),
        "from_core": len(core),
        "from_openreview": len(openreview),
        "from_github": len(github),
        "from_zenodo": len(zenodo),
        "from_opencitations": len(opencitations),
        "from_springer": len(springer),
        "from_ieee_xplore": len(ieee_xplore),
        "from_figshare": len(figshare),
        "from_openml": len(openml),
        "from_gdelt": len(gdelt),
        "from_wikidata": len(wikidata),
        "from_orcid": len(orcid),
        "source_errors": errors,
        "missing_pdf_url": missing_pdf_url,
        "download_http_error": download_http_error,
        "non_pdf_content_type": non_pdf_content_type,
        "blocked_or_paywalled": blocked_or_paywalled,
        "require_pdf_filtered": require_pdf_filtered,
        "sources_config_applied": int(bool(sources_cfg)),
        "max_total_results": max_total_results,
        "max_pdf_downloads": max_pdf_downloads,
        "unsupported_enabled_sources_count": len(unsupported),
        "unsupported_enabled_sources": unsupported,
        "unpaywall_enabled": int(unpaywall_enabled),
        "unpaywall_lookups": unpaywall_lookups,
        "unpaywall_enriched": unpaywall_enriched,
        "unpaywall_lookup_errors": unpaywall_lookup_errors,
        "unpaywall_cache_hits": unpaywall_cache_hits,
        "unpaywall_cache_misses": unpaywall_cache_misses,
        "unpaywall_cache_path": unpaywall_cache_path if unpaywall_enabled else None,
    }
    source_quality_healthy, source_quality_reasons, source_quality_metrics = _evaluate_source_quality(stats, sources_cfg)
    stats["source_quality_healthy"] = source_quality_healthy
    stats["source_quality_reasons"] = source_quality_reasons
    stats["source_quality_metrics"] = source_quality_metrics
    stats["source_quality_thresholds"] = ((sources_cfg.get("limits", {}) or {}).get("source_quality", {}))
    return stats
