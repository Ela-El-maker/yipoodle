from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


_ALL_SOURCES: dict[str, dict[str, Any]] = {
    "arxiv": {"enabled": False, "endpoint": "http://export.arxiv.org/api/query", "max_results": 40},
    "openalex": {"enabled": False, "endpoint": "https://api.openalex.org/works", "max_results": 30},
    "semanticscholar": {
        "enabled": False,
        "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
        "max_results": 30,
    },
    "crossref": {"enabled": False, "endpoint": "https://api.crossref.org/works", "max_results": 20},
    "dblp": {"enabled": False, "endpoint": "https://dblp.org/search/publ/api", "max_results": 20},
    "paperswithcode": {"enabled": False, "endpoint": "https://paperswithcode.com/api/v1/papers/"},
    "core": {"enabled": False, "endpoint": "https://api.core.ac.uk/v3/search/works", "max_results": 20},
    "openreview": {"enabled": False, "endpoint": "https://api2.openreview.net/notes", "max_results": 20},
    "github": {
        "enabled": False,
        "endpoint": "https://api.github.com/search/repositories",
        "max_results": 15,
        "auth": {"header": "Authorization", "value": "Bearer ${GITHUB_TOKEN}"},
    },
    "zenodo": {"enabled": False, "endpoint": "https://zenodo.org/api/records", "max_results": 15},
    "opencitations": {"enabled": False, "endpoint": "https://opencitations.net/index/api/v1/metadata", "max_results": 20},
    "springer": {
        "enabled": False,
        "endpoint": "https://api.springernature.com/metadata/json",
        "max_results": 20,
        "auth": {"query_param": "api_key", "value": "${SPRINGER_API_KEY}"},
    },
    "ieee_xplore": {
        "enabled": False,
        "endpoint": "https://ieeexploreapi.ieee.org/api/v1/search/articles",
        "max_results": 20,
        "auth": {"query_param": "apikey", "value": "${IEEE_API_KEY}"},
    },
    "figshare": {"enabled": False, "endpoint": "https://api.figshare.com/v2/articles", "max_results": 20},
    "openml": {"enabled": False, "endpoint": "https://www.openml.org/api/v1/json", "max_results": 20},
    "gdelt": {"enabled": False, "endpoint": "https://api.gdeltproject.org/api/v2/doc/doc", "max_results": 15},
    "wikidata": {"enabled": False, "endpoint": "https://query.wikidata.org/sparql", "max_results": 20},
    "orcid": {"enabled": False, "endpoint": "https://pub.orcid.org/v3.0/expanded-search/", "max_results": 20},
    "unpaywall": {
        "enabled": False,
        "endpoint": "https://api.unpaywall.org/v2",
        "required_query_params": {"email": "${UNPAYWALL_EMAIL}"},
    },
}


_PROFILES: dict[str, list[str]] = {
    "academic": ["arxiv", "openalex", "semanticscholar", "crossref", "unpaywall"],
    "industry": ["openalex", "crossref", "github", "zenodo", "gdelt", "unpaywall"],
    "balanced": ["arxiv", "openalex", "semanticscholar", "crossref", "github", "unpaywall"],
}


def _slug(value: str) -> str:
    v = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    v = re.sub(r"_+", "_", v).strip("_")
    return v or "custom_domain"


def _guess_profile(domain_slug: str, profile: str) -> str:
    if profile != "auto":
        return profile
    if any(k in domain_slug for k in ("marketing", "growth", "business", "market")):
        return "industry"
    return "balanced"


def scaffold_domain_config(domain: str, out_path: str | None = None, profile: str = "auto", overwrite: bool = False) -> str:
    domain_slug = _slug(domain)
    chosen = _guess_profile(domain_slug, profile)
    enabled = set(_PROFILES.get(chosen, _PROFILES["balanced"]))

    sources = {k: dict(v) for k, v in _ALL_SOURCES.items()}
    for name, cfg in sources.items():
        cfg["enabled"] = name in enabled

    doc: dict[str, Any] = {
        "domain": domain_slug,
        "description": f"Auto-generated domain config for {domain_slug}.",
        "sources": sources,
        "ocr": {
            "enabled": False,
            "timeout_sec": 30,
            "min_chars_trigger": 120,
            "max_pages": 20,
            "min_output_chars": 200,
            "min_gain_chars": 40,
            "min_confidence": 45.0,
            "lang": "eng",
            "profile": "document",
            "noise_suppression": True,
        },
        "limits": {
            "default_max_results": 25,
            "max_total_results": 150,
            "max_pdf_downloads": 20,
            "max_tokens_per_summary": 1200,
            "source_quality": {
                "max_source_error_rate": 0.5,
                "min_download_success_rate": 0.1,
                "max_download_http_error_rate": 0.8,
                "max_non_pdf_content_rate": 0.95,
                "max_blocked_or_paywalled_rate": 0.95,
                "max_unpaywall_lookup_error_rate": 0.95,
            },
        },
        "ranking": {
            "strategy": "hybrid",
            "weights": {
                "recency": 0.30,
                "citation_count": 0.30,
                "semantic_similarity": 0.30,
                "source_trust": 0.10,
            },
        },
    }

    dst = Path(out_path) if out_path else Path("config/domains") / f"sources_{domain_slug}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {dst}. Use --overwrite to replace it.")

    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return str(dst)
