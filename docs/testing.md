# Testing

This document describes Yipoodle's test suite, how to run tests, the test structure, and guidelines for writing new tests.

---

## Quick Start

```bash
# Run all tests
pytest -q

# Run with Make
make test

# Run a specific test file
pytest tests/test_retrieval_priors.py -v

# Run a specific test function
pytest tests/test_query_router.py::test_arithmetic_route -v

# Run with output
pytest -s
```

---

## Test Suite Overview

The test suite contains **254 tests** across **85+ test files**, covering every major subsystem. All tests run fully offline with no external API calls.

### Test Counts by Subsystem

| Subsystem                              | Test Files                                                                                                                                                                                                                                                                           | Coverage                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Tiny GPT (model, training, generation) | `test_shapes.py`, `test_save_load.py`, `test_tokenizer.py`, `test_dataset.py`                                                                                                                                                                                                        | Model forward/backward shapes, checkpoint save/load, tokenizer encode/decode, dataset batching |
| Research Pipeline                      | `test_integration.py`, `test_apps.py`, `test_research_*.py`                                                                                                                                                                                                                          | End-to-end pipeline, research copilot, retrieval modes, direct answer, live merge              |
| Retrieval & Indexing                   | `test_hybrid_retrieval.py`, `test_retrieval_priors.py`, `test_retrieval_quality_prior.py`, `test_index_enrichment.py`, `test_index_health_gate.py`, `test_build_index_live_data.py`                                                                                                  | BM25 scoring, hybrid fusion, metadata priors, quality weighting, index building                |
| Vector Search                          | `test_vector_index.py`, `test_vector_backend.py`, `test_vector_sharding.py`, `test_vector_service.py`, `test_vector_service_http.py`, `test_vector_eval_gate.py`                                                                                                                     | FAISS indexing, sharding, ANN backends, HTTP service, recall evaluation                        |
| Paper Search & Sync                    | `test_paper_search.py`, `test_paper_search_retry.py`, `test_paper_sync_categories.py`                                                                                                                                                                                                | Source connectors, retry logic, sync deduplication                                             |
| Extraction                             | `test_extraction_fallback.py`, `test_extraction_eval.py`, `test_extraction_quality_report.py`, `test_extraction_reason_stats.py`, `test_extract_corpus_sources_ocr.py`, `test_two_column_extraction.py`, `test_ocr_fallback.py`, `test_ocr_lang_auto.py`                             | Extractor fallback chain, quality reports, OCR, two-column, layout engine                      |
| Layout Engine                          | `test_layout_engine_v2.py`, `test_layout_promotion_gate.py`                                                                                                                                                                                                                          | v2 layout regions, promotion gate logic                                                        |
| Validation                             | `test_validators.py`, `test_faithfulness.py`, `test_semantic_validation.py`, `test_live_citation_validation.py`                                                                                                                                                                      | Citation format, fabrication detection, semantic faithfulness, claim support                   |
| Knowledge Base                         | `test_kb_store.py`, `test_kb_claim_parser.py`, `test_kb_confidence.py`, `test_kb_contradictions.py`, `test_kb_diff.py`, `test_kb_ingest_validated_report.py`, `test_kb_backfill_last_n.py`                                                                                           | Store CRUD, claim parsing, confidence decay, contradiction detection, diffs, ingestion         |
| Query Router                           | `test_query_router.py`, `test_query_router_eval.py`, `test_cli_query_dispatch.py`, `test_cli_retrieval_modes.py`, `test_cli_semantic_flags.py`                                                                                                                                       | Routing logic, eval harness, CLI dispatch                                                      |
| Ask Mode                               | `test_ask_mode.py`, `test_direct_answer.py`                                                                                                                                                                                                                                          | Arithmetic, unit conversion, glossary, definition fallback                                     |
| Monitor Mode                           | `test_monitor_mode.py`, `test_monitoring_*.py`, `test_cli_monitor*.py`                                                                                                                                                                                                               | Monitor creation, rules, state, hooks, soak simulation, history                                |
| Notes Mode                             | `test_notes_mode.py`                                                                                                                                                                                                                                                                 | Research + KB ingest + notes generation                                                        |
| Automation                             | `test_automation.py`, `test_automation_alerts.py`, `test_automation_monitoring_integration.py`                                                                                                                                                                                       | Run orchestration, alert dispatch, monitoring integration                                      |
| Live Sources                           | `test_live_sources_config.py`, `test_live_fetch_rest.py`, `test_live_fetch_rss.py`, `test_live_routing.py`, `test_live_snapshot_store.py`, `test_live_failure_fallback.py`                                                                                                           | Config parsing, REST/RSS fetch, routing, snapshots                                             |
| Pipeline                               | `test_pipeline_runner.py`                                                                                                                                                                                                                                                            | End-to-end pipeline gating                                                                     |
| Corpus                                 | `test_corpus_health.py`, `test_corpus_migration.py`                                                                                                                                                                                                                                  | Health evaluation, metadata migration                                                          |
| Other                                  | `test_ids.py`, `test_schemas.py`, `test_sources_config.py`, `test_snapshot.py`, `test_cache_and_perf.py`, `test_benchmark_scale.py`, `test_domain_scaffold.py`, `test_shortlist_metadata.py`, `test_quality_scoring.py`, `test_query_builder.py`, `test_relevance_reject_reasons.py` | ID normalization, schema validation, config loading, caching, benchmarking                     |

---

## Test Configuration

### pytest.ini

```ini
[pytest]
pythonpath = .
```

This ensures the project root is on `sys.path` so `from src.xxx import ...` works without installation.

### Test Fixtures

Test fixtures are stored in `tests/fixtures/`:

- `extraction_gold.json` — gold-standard checks for extraction evaluation.
- `queries.txt` — benchmark query set.
- `router_eval_cases.json` — labeled routing test cases.
- `extracted/` — sample extracted corpus for testing.

---

## Testing Patterns

### Mocking External APIs

Tests mock all HTTP calls. No external API is contacted during testing:

```python
from unittest.mock import patch, MagicMock

@patch("src.apps.paper_search.requests.get")
def test_search_arxiv(mock_get):
    mock_get.return_value = MagicMock(status_code=200, text="<xml>...</xml>")
    results = search_arxiv("transformer")
    assert len(results) > 0
```

### Testing with Temporary Files

Many tests use `tmp_path` (pytest fixture) to create temporary directories:

```python
def test_build_index(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Create test snippets...
    result = build_index(str(corpus_dir), str(tmp_path / "index.json"))
    assert result["snippet_count"] > 0
```

### Testing CLI Commands

CLI tests invoke subcommands through the argument parser rather than shell calls:

```python
def test_cli_ask(capsys):
    from src.cli import main
    sys.argv = ["cli.py", "ask", "--question", "2 + 2"]
    main()
    captured = capsys.readouterr()
    assert "4" in captured.out
```

---

## Running Specific Test Categories

```bash
# Model tests only
pytest tests/test_shapes.py tests/test_save_load.py tests/test_tokenizer.py tests/test_dataset.py -v

# Research pipeline tests
pytest tests/test_integration.py tests/test_apps.py -v

# Validation tests
pytest tests/test_validators.py tests/test_faithfulness.py tests/test_semantic_validation.py -v

# KB tests
pytest tests/test_kb_store.py tests/test_kb_claim_parser.py tests/test_kb_confidence.py -v

# All monitoring tests
pytest tests/test_monitor_mode.py tests/test_monitoring_*.py -v

# Vector search tests
pytest tests/test_vector_*.py -v
```

---

## Writing New Tests

### Conventions

1. **File naming**: `tests/test_<module_name>.py` — mirrors the source module.
2. **Function naming**: `test_<behavior_description>` — descriptive of what's being verified.
3. **No external dependencies**: All tests must work offline. Mock HTTP calls.
4. **Use tmp_path**: For any file I/O, use the `tmp_path` pytest fixture.
5. **Assert specifics**: Check exact values, not just truthiness.

### Template

```python
"""Tests for src/apps/my_module.py"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.apps.my_module import my_function


def test_basic_behavior(tmp_path):
    """my_function returns expected result for valid input."""
    result = my_function(input_data="test")
    assert result["status"] == "ok"
    assert result["count"] > 0


def test_edge_case():
    """my_function handles empty input gracefully."""
    result = my_function(input_data="")
    assert result["status"] == "empty"


@patch("src.apps.my_module.requests.get")
def test_with_mocked_http(mock_get):
    """my_function handles API response correctly."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"data": [{"id": "1", "title": "Test"}]},
    )
    result = my_function(query="test")
    assert len(result) == 1
```

---

## CI Integration

Tests are designed for CI pipelines:

```bash
# Standard CI command
python -m pytest -q --tb=short

# With JUnit XML output
python -m pytest --junitxml=reports/junit.xml

# With coverage (requires pytest-cov)
python -m pytest --cov=src --cov-report=html
```

All 254 tests complete in ~12 seconds on a standard machine.
