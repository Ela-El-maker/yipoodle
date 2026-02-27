# System Overview

This document explains the purpose, design philosophy, and key concepts behind Yipoodle.

---

## What Is Yipoodle?

Yipoodle is an offline research automation platform that combines two capabilities:

1. **Research Copilot** — an automated pipeline that discovers academic papers from 18+ sources, extracts their text, builds searchable indexes, synthesizes evidence-grounded reports, and validates every claim against the source material.

2. **Tiny GPT Engine** — a from-scratch character-level GPT transformer for local text generation, built as a learning-first implementation with no dependency on Hugging Face model classes.

The system is designed for researchers, engineers, and teams who need:

- Automated literature surveys with full citation traceability.
- Quality-gated research pipelines that catch fabricated or unsupported claims.
- Scheduled monitoring of research topics with alerting.
- A reproducible, auditable research workflow.

---

## Design Philosophy

### 1. Deterministic, Not Generative

Unlike LLM-based research tools, Yipoodle does not use any language model for report synthesis. All output text is derived directly from source snippets through term overlap, TF-IDF scoring, and template rendering. This means:

- Every sentence in a report traces to a specific evidence snippet.
- No hallucinated facts, fabricated citations, or invented numbers.
- Output quality depends on input corpus quality, not model capability.

### 2. Offline-First

After the initial paper sync (which fetches papers from online APIs), the entire pipeline runs offline:

- PDF extraction, indexing, retrieval, synthesis, and validation require no internet.
- The system works on air-gapped machines or consumer laptops.
- No cloud API keys required for core functionality.

### 3. Multi-Stage Quality Gating

Every pipeline stage has a quality gate that can block or warn:

```
Source Quality Gate ─→ Corpus Health Gate ─→ Relevance Gate ─→ Citation Gate ─→ Faithfulness Gate
```

- **Source quality**: error rates, download success, paywall detection.
- **Corpus health**: minimum snippets, chars per paper, extraction error rate.
- **Relevance**: question-evidence similarity threshold.
- **Citation**: format validation, coverage metrics.
- **Faithfulness**: semantic similarity, contradiction detection, fabrication checks.

### 4. Shadow Mode Rollout

New features deploy in shadow mode first — they execute and log results but don't block the pipeline. Once validated, they promote to enforced mode. Examples:

- Semantic validation started in shadow mode before becoming enforceable.
- Layout engine v2 runs in shadow mode alongside legacy, with a promotion gate for controlled switching.

### 5. Reproducibility

Every research run produces:

- A markdown report with inline citations.
- An evidence JSON file with all retrieved snippets.
- A metrics JSON file with coverage and quality statistics.
- Optionally, a snapshot bundle containing configs, indexes, and artifacts for exact replay.

---

## Key Concepts

### Paper Records

A `PaperRecord` represents metadata about an academic paper:

- Title, authors, abstract, year, venue.
- DOI, arXiv ID, PDF URL.
- Citation count, source provenance.

Papers are stored in a SQLite database (`data/papers.db`) and deduplicated by DOI or title hash.

### Snippets

A `SnippetRecord` is a chunk of extracted text from a PDF:

- Paper ID, snippet index, text content.
- Page number, section label (introduction, methods, results, etc.).
- Extraction quality score and quality band (good/fair/poor).
- Extraction metadata (extractor used, char count, OCR status).

### Evidence

An `EvidenceItem` is a scored retrieval result:

- Snippet reference with relevance score.
- Retrieval mode (lexical/vector/hybrid).
- Score components (BM25, cosine similarity, metadata priors).

### Evidence Pack

An `EvidencePack` bundles a question with its retrieved evidence items — the input to report synthesis.

### Research Report

A `ResearchReport` contains:

- **Synthesis**: evidence-grounded prose with inline citations `(P<paper_id>:S<snippet_n>)`.
- **Key Claims**: extracted assertions from the synthesis.
- **Research Gaps**: identified areas lacking evidence.
- **Experiment Proposals**: suggested follow-up experiments.
- **Shortlist**: ranked papers with relevance reasons.
- **Diagnostics**: retrieval metadata, scoring details, intent routing.

### Knowledge Base

A long-lived SQLite store for accumulating knowledge across runs:

- **Claims** are versioned and indexed with FTS5 for fast search.
- **Confidence** decays over time (configurable daily rate).
- **Contradictions** are auto-detected between claims via negation overlap.
- **Topics** provide namespace isolation (e.g., "finance_markets", "computer_vision").

### Query Modes

| Mode     | Purpose                                                  | Corpus Required                |
| -------- | -------------------------------------------------------- | ------------------------------ |
| ASK      | Quick deterministic answers (math, conversion, glossary) | No                             |
| RESEARCH | Full evidence-grounded report                            | Yes                            |
| MONITOR  | Create recurring topic watch with cron scheduling        | No (creates automation config) |
| NOTES    | Research + KB ingest + structured study notes            | Yes                            |

### Automation Workflow Extensions

The automation layer also supports operational workflows beyond the core run:

- **Research templates** (`research-template`) for repeatable multi-query session packs.
- **Source reliability watchdog** (`reliability-watchdog`) for connector health scoring.
- **Benchmark regression gate** (`benchmark-regression-check`) for latency/quality drift detection.
- **File-drop ingestion watcher** (`watch-ingest`) for incremental updates when new PDFs appear.

---

## Intended Users

| User Type       | Primary Use Case                                                       |
| --------------- | ---------------------------------------------------------------------- |
| **Researchers** | Automated literature surveys, evidence synthesis, gap identification   |
| **Engineers**   | Technical research on specific topics, monitoring for new developments |
| **Teams**       | Scheduled research updates with alerting, shared knowledge base        |
| **Students**    | Study notes from research papers, concept exploration                  |

---

## What Yipoodle Is NOT

- **Not an open-ended agent**: v1 chat UI is deterministic mode-dispatch over ASK/QUERY/RESEARCH/NOTES/MONITOR/AUTOMATION, not autonomous multi-agent planning.
- **Not a search engine**: It synthesizes and validates, not just retrieves.
- **Not an LLM wrapper**: No generative model is used for report content. Everything is deterministic.
- **Not a paper manager**: It doesn't replace Zotero or Mendeley. It automates the research pipeline from discovery to validated report.

---

## System Boundaries

### What Requires Internet

- `sync-papers`: fetching paper metadata from APIs.
- `sync-papers`: downloading PDFs.
- `live-fetch`: fetching live data from REST/RSS sources.
- Online semantic validation (optional).
- Unpaywall enrichment (optional).

### What Works Offline

- PDF extraction and text processing.
- Index building (BM25 + FAISS).
- Retrieval and scoring.
- Report synthesis and validation.
- Knowledge base operations.
- Tiny GPT training and generation.
- All tests (254 tests run fully offline).
