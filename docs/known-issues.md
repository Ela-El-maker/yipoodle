# Known Issues & Improvement Opportunities

This document tracks known limitations, architectural trade-offs, and potential improvements for Yipoodle.

---

## Known Limitations

### 1. Deterministic Synthesis Quality

**Issue**: Report synthesis is built entirely from term overlap and template rendering — no language model generates the prose. This guarantees traceability but produces repetitive, mechanical text.

**Impact**: Reports are accurate and fully cited but less readable than LLM-generated summaries.

**Mitigation**: The design is intentional — traceability over fluency. An optional LLM post-processing layer could be added in the future with grounding constraints.

### 2. Subprocess Overhead in Automation

**Issue**: The automation engine (`src/apps/automation.py`) runs each pipeline stage as a separate `python -m src.cli` subprocess via `subprocess.run()`. This provides process isolation but adds ~0.5-1s startup overhead per stage.

**Impact**: A 3-topic automation run with 6 stages each incurs ~18 subprocess launches. Total overhead is ~10-15 seconds.

**Mitigation**: Intentional design trade-off for fault isolation. Could be optimized with in-process function calls for trusted environments.

### 3. Large Parameter Surfaces

**Issue**: The `run_research()` function accepts ~50 parameters. The cache key (`_cache_key()`) hashes ~40 of them. Adding new parameters requires updating the cache key to avoid stale cache hits.

**Impact**: Risk of cache key drift when new features are added.

**Mitigation**: Systematic review of cache key composition when adding new retrieval parameters.

### 4. Hardcoded Domain References

**Issue**: Some code paths reference "computer_vision" as a default domain and `query_builder.py` contains domain-specific query expansion terms.

**Impact**: Domain expansion for non-CV topics requires explicit configuration via `--sources-config`.

**Mitigation**: Use domain-specific source configs (`config/domains/`) to override defaults. The `scaffold-domain-config` command generates templates for new domains.

### 5. PapersWithCode Fallback

**Issue**: The PapersWithCode connector silently falls back to the HuggingFace API when the PapersWithCode endpoint is unreachable. The HuggingFace API returns a different data shape.

**Impact**: Results may differ between primary and fallback paths without explicit indication to the user.

**Mitigation**: Fallback is logged in source fetch stats. Consider adding an explicit warning flag.

### 6. Gmail SMTP Credentials

**Issue**: Email alerting uses `GMAIL_APP_PASSWORD` from environment variables. The email recipient addresses are stored in plaintext in `config/automation.yaml`.

**Impact**: Credential management relies on environment variable discipline. Not suitable for shared or multi-tenant deployments without additional secrets management.

**Mitigation**: Use environment variables for sensitive credentials; never commit `GMAIL_APP_PASSWORD` to version control.

### 7. No Incremental Index Updates

**Issue**: Index building always rebuilds from scratch. There's no incremental update path for adding new papers to an existing index.

**Impact**: Rebuilding a large index with vector embeddings can be slow (~minutes for 10k+ snippets).

**Mitigation**: Planned improvement. For now, keep corpus sizes manageable per topic.

### 8. Single-Process Vector Service

**Issue**: The vector HTTP service (`vector-service-serve`) uses Python's `ThreadingHTTPServer`, which is single-process and not production-grade.

**Impact**: Suitable for development and single-user use only. Cannot handle concurrent load.

**Mitigation**: For production, deploy behind a reverse proxy or migrate to a proper ASGI server.

---

## Architectural Improvement Opportunities

### Short-Term

1. **Add `--verbose` / `--quiet` flags** to CLI for controlling output verbosity globally.
2. **Structured logging**: Replace `print()` statements with Python `logging` module for configurable log levels.
3. **Parallel source fetching**: `sync-papers` could fetch from multiple sources concurrently using `concurrent.futures`.
4. **Index update mode**: Support appending new snippets to an existing index without full rebuild.

### Medium-Term

5. **Configuration validation**: Add Pydantic models for YAML config files to catch misconfigurations early with clear error messages.
6. **Plugin architecture for sources**: Make source connectors pluggable so new sources can be added without modifying `paper_search.py`.
7. **Web UI**: Add a simple web interface for browsing reports, searching the KB, and monitoring status.
8. **Report diffing**: Compare two research reports on the same topic to show what changed.

### Long-Term

9. **Optional LLM integration**: Add an optional LLM post-processing layer for report polishing while maintaining citation grounding constraints.
10. **Multi-user support**: Add user authentication and tenant isolation for shared deployments.
11. **Streaming pipeline**: Replace batch processing with a streaming architecture for real-time paper ingestion and indexing.

---

## Resolved Issues

| Issue                        | Resolution                                                           | Date     |
| ---------------------------- | -------------------------------------------------------------------- | -------- |
| Checkpoint loading security  | Hardened to `weights_only=True` with strict schema validation        | Resolved |
| Missing extraction metadata  | `migrate-extraction-meta` command backfills legacy files             | Resolved |
| Layout engine reliability    | Shadow mode deployment allows safe rollout of v2 with promotion gate | Resolved |
| OCR quality control          | Added confidence thresholds, gain checks, and noise suppression      | Resolved |
| Source quality blindness     | Added per-source quality metrics and configurable gate thresholds    | Resolved |
| Live data citation integrity | Live evidence uses `SNAP:<id>:S<n>` IDs with snapshot persistence    | Resolved |
