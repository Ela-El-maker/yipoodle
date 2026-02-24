# CLI Reference

Yipoodle provides ~40 CLI subcommands accessed via `python -m src.cli <command>`. This document lists every command grouped by category.

---

## Quick Reference

| Category                | Commands                                                                                                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Research Pipeline**   | `sync-papers`, `build-index`, `research`, `run-pipeline`, `validate-report`, `benchmark-research`, `benchmark-scale`                                                   |
| **Tiny GPT**            | `train-tinygpt`, `generate`                                                                                                                                            |
| **Query Router**        | `ask`, `query`, `notes`, `query-router-eval`, `live-fetch`                                                                                                             |
| **Knowledge Base**      | `kb-ingest`, `kb-query`, `kb-diff`, `kb-backfill`                                                                                                                      |
| **Monitoring**          | `monitor`, `monitor-unregister`, `monitor-evaluate`, `monitor-digest-flush`, `monitor-status`, `monitor-soak-sim`, `monitor-history-check`                             |
| **Vector Service**      | `vector-service-build`, `vector-service-query`, `vector-service-health`, `vector-service-serve`                                                                        |
| **Corpus & Extraction** | `extract-corpus`, `corpus-health`, `extraction-quality-report`, `extraction-eval`, `extraction-eval-scaffold-gold`, `layout-promotion-gate`, `migrate-extraction-meta` |
| **Utility**             | `snapshot-run`, `doc-write`, `release-notes`, `scaffold-domain-config`                                                                                                 |

---

## Research Pipeline

### `sync-papers`

Fetch paper metadata from academic APIs and download PDFs.

```bash
python -m src.cli sync-papers --query "transformer attention" --max-results 20 --with-semantic-scholar
```

| Argument                        | Default               | Description                                |
| ------------------------------- | --------------------- | ------------------------------------------ |
| `--query`                       | _(required)_          | Search query                               |
| `--max-results`                 | 20                    | Maximum papers to fetch                    |
| `--db-path`                     | `data/papers.db`      | SQLite database path                       |
| `--papers-dir`                  | `data/papers`         | PDF download directory                     |
| `--with-semantic-scholar`       | false                 | Enable Semantic Scholar source             |
| `--prefer-arxiv`                | false                 | Prefer arXiv versions                      |
| `--require-pdf`                 | false                 | Skip papers without PDF URLs               |
| `--sources-config`              | `config/sources.yaml` | Sources configuration file                 |
| `--fail-on-source-quality-gate` | false                 | Exit non-zero if source quality gate fails |

### `build-index`

Build BM25 lexical index (and optional FAISS vector index) from extracted corpus.

```bash
python -m src.cli build-index --corpus data/extracted --out data/indexes/bm25_index.json \
  --db-path data/papers.db --require-healthy-corpus
```

| Argument                   | Default                        | Description                               |
| -------------------------- | ------------------------------ | ----------------------------------------- |
| `--corpus`                 | `data/extracted`               | Extracted corpus directory                |
| `--out`                    | `data/indexes/bm25_index.json` | Output index path                         |
| `--db-path`                | None                           | SQLite DB for metadata enrichment         |
| `--with-vector`            | false                          | Also build FAISS vector index             |
| `--embedding-model`        | `all-MiniLM-L6-v2`             | Sentence-transformer model                |
| `--vector-index-type`      | `flat`                         | FAISS backend: `flat`, `ivf_flat`, `hnsw` |
| `--vector-shards`          | 1                              | Number of index shards                    |
| `--require-healthy-corpus` | false                          | Enforce corpus health gate                |
| `--live-data`              | []                             | Live data JSON to merge (repeatable)      |

### `research`

Generate an evidence-grounded research report from an index.

```bash
python -m src.cli research --index data/indexes/bm25_index.json \
  --question "What are current limitations?" --top-k 8 \
  --out runs/research_reports/report.md
```

| Argument                 | Default                           | Description                              |
| ------------------------ | --------------------------------- | ---------------------------------------- |
| `--index`                | `data/indexes/bm25_index.json`    | BM25 index path                          |
| `--question`             | _(required)_                      | Research question                        |
| `--top-k`                | 8                                 | Number of evidence items                 |
| `--min-items`            | 2                                 | Minimum required evidence items          |
| `--min-score`            | 0.5                               | Minimum relevance score threshold        |
| `--retrieval-mode`       | `lexical`                         | `lexical`, `vector`, or `hybrid`         |
| `--alpha`                | 0.6                               | Hybrid fusion weight (lexical vs vector) |
| `--max-per-paper`        | 2                                 | Max snippets per paper                   |
| `--quality-prior-weight` | 0.15                              | Extraction quality score weight          |
| `--live`                 | false                             | Enable live data sources                 |
| `--live-merge-mode`      | `union`                           | `union` or `live_first`                  |
| `--use-kb`               | false                             | Merge KB candidates into retrieval       |
| `--no-cache`             | false                             | Force recomputation                      |
| `--out`                  | `runs/research_reports/report.md` | Output report path                       |

Outputs: `report.md`, `report.evidence.json`, `report.json`, `report.metrics.json`

### `run-pipeline`

One-command end-to-end pipeline: sync → extract → health → index → research → validate.

```bash
python -m src.cli run-pipeline --query "mobile segmentation limitations" \
  --sources-config config/sources.yaml --with-semantic-scholar
```

| Argument                        | Default                                      | Description                 |
| ------------------------------- | -------------------------------------------- | --------------------------- |
| `--query`                       | _(required)_                                 | Research question           |
| `--max-results`                 | 20                                           | Papers to fetch             |
| `--require-healthy-corpus`      | true                                         | Enforce corpus health gate  |
| `--fail-on-source-quality-gate` | true                                         | Enforce source quality gate |
| `--dispatch-alerts`             | false                                        | Dispatch alerts on failure  |
| `--summary-out`                 | `runs/research_reports/pipeline_latest.json` | Pipeline summary output     |

### `validate-report`

Validate report citations, check for fabrication, and run semantic faithfulness scoring.

```bash
python -m src.cli validate-report --input runs/research_reports/report.md \
  --evidence runs/research_reports/report.evidence.json
```

| Argument                         | Default      | Description                         |
| -------------------------------- | ------------ | ----------------------------------- |
| `--input`                        | _(required)_ | Report markdown path                |
| `--evidence`                     | _(required)_ | Evidence JSON path                  |
| `--semantic-mode`                | `offline`    | `offline`, `online`, or `hybrid`    |
| `--semantic-shadow-mode`         | true         | Log but don't block on failures     |
| `--semantic-fail-on-low-support` | false        | Exit non-zero on low support scores |

### `benchmark-research`

Benchmark research runtime and cache effects.

```bash
python -m src.cli benchmark-research --index data/indexes/bm25_index.json \
  --queries-file tests/fixtures/queries.txt --runs-per-query 3
```

### `benchmark-scale`

Benchmark retrieval on synthetically enlarged corpus.

```bash
python -m src.cli benchmark-scale --corpus data/extracted \
  --queries-file tests/fixtures/queries.txt --repeat-factor 50
```

---

## Tiny GPT

### `train-tinygpt`

Train the character-level GPT model.

```bash
python -m src.cli train-tinygpt --data data/data.txt --config config/train.yaml
```

| Argument    | Default         | Description                           |
| ----------- | --------------- | ------------------------------------- |
| `--data`    | `data/data.txt` | Training text file                    |
| `--out-dir` | `runs`          | Output directory for checkpoints/logs |
| `--config`  | None            | YAML config file path                 |

### `generate`

Generate text from a trained checkpoint.

```bash
python -m src.cli generate --checkpoint runs/checkpoints/step_1000.pt \
  --prompt "The future of" --max-new-tokens 200
```

| Argument           | Default      | Description          |
| ------------------ | ------------ | -------------------- |
| `--checkpoint`     | _(required)_ | Checkpoint file path |
| `--prompt`         | _(required)_ | Text prompt          |
| `--max-new-tokens` | 200          | Tokens to generate   |
| `--temperature`    | 0.9          | Sampling temperature |
| `--top-k`          | 40           | Top-k filtering      |
| `--deterministic`  | false        | Use greedy decoding  |
| `--seed`           | 1337         | Random seed          |

---

## Query Router

### `query`

Smart prompt dispatcher — routes to ask/research/monitor/notes automatically.

```bash
python -m src.cli query --question "What is overfitting?"
python -m src.cli query --question "Compare ViT and CNN" --index data/indexes/bm25_index.json
python -m src.cli query --mode research --question "..." --index data/indexes/bm25_index.json
```

| Argument     | Default      | Description                                               |
| ------------ | ------------ | --------------------------------------------------------- |
| `--question` | _(required)_ | User prompt                                               |
| `--mode`     | `auto`       | Force mode: `auto`, `ask`, `research`, `monitor`, `notes` |

Inherits all arguments from `research`, `ask`, `monitor`, and `notes` modes.

### `ask`

Quick deterministic answer mode (arithmetic, conversion, glossary).

```bash
python -m src.cli ask --question "45 km/h to m/s"
python -m src.cli ask --question "What is an algorithm?"
```

| Argument     | Default                    | Description   |
| ------------ | -------------------------- | ------------- |
| `--question` | _(required)_               | Question text |
| `--config`   | `config/router.yaml`       | Router config |
| `--glossary` | `config/ask_glossary.yaml` | Glossary file |

### `notes`

Research + KB ingest + structured study notes.

```bash
python -m src.cli notes --question "Create study notes on attention mechanisms" \
  --index data/indexes/bm25_index.json
```

### `query-router-eval`

Evaluate router accuracy against a labeled test set.

```bash
python -m src.cli query-router-eval --cases tests/fixtures/router_eval_cases.json \
  --strict-min-accuracy 0.95
```

### `live-fetch`

Fetch data from a configured live source.

```bash
python -m src.cli live-fetch --source yahoo_finance --query "NVDA" \
  --params symbol=NVDA --sources-config config/sources.yaml
```

---

## Knowledge Base

### `kb-ingest`

Ingest validated report claims into the knowledge base.

```bash
python -m src.cli kb-ingest --report runs/research_reports/report.md \
  --evidence runs/research_reports/report.evidence.json \
  --kb-db data/kb/knowledge.db --topic finance_markets
```

### `kb-query`

Search the knowledge base.

```bash
python -m src.cli kb-query --kb-db data/kb/knowledge.db \
  --query "transformer drift risk" --topic finance_markets --top-k 10
```

### `kb-diff`

Show KB changes since a run ID or timestamp.

```bash
python -m src.cli kb-diff --kb-db data/kb/knowledge.db \
  --topic finance_markets --since-run 20260224T120000Z
```

### `kb-backfill`

Ingest the last N reports into the KB.

```bash
python -m src.cli kb-backfill --kb-db data/kb/knowledge.db \
  --reports-dir runs/research_reports --topic finance_markets --last-n 20
```

---

## Monitoring

### `monitor`

Create a recurring topic watch and run a baseline.

```bash
python -m src.cli monitor --question "Track PIX outage signals"
python -m src.cli monitor --question "..." --schedule-backend file --no-register-schedule
```

### `monitor-unregister`

Remove a monitor's cron registration and optionally delete generated files.

```bash
python -m src.cli monitor-unregister --name track_pix_outage_signals --delete-files
```

### `monitor-status`

Show monitoring state, digest queue, and schedule registry.

```bash
python -m src.cli monitor-status --config config/automation.yaml
```

### `monitor-soak-sim`

Simulate multi-day monitoring behavior for cooldown/hysteresis tuning.

```bash
python -m src.cli monitor-soak-sim --topic finance_risk --runs 96 \
  --interval-minutes 60 --cooldown-minutes 360
```

### `monitor-history-check`

Analyze real monitoring trigger history from audit files.

```bash
python -m src.cli monitor-history-check --topic cv_mobile_segmentation_limits \
  --audit-dir runs/smoke/monitoring/audit
```

---

## Vector Service

### `vector-service-build`

Build vector sidecars from a lexical index.

```bash
python -m src.cli vector-service-build --index data/indexes/bm25_index.json \
  --vector-index-type hnsw --vector-m 32
```

### `vector-service-serve`

Start local vector HTTP service.

```bash
python -m src.cli vector-service-serve --index data/indexes/bm25_index.json \
  --host 127.0.0.1 --port 8765
```

Endpoints: `GET /health`, `POST /query`

### `vector-service-query`

Query vector sidecars directly (debugging).

```bash
python -m src.cli vector-service-query --index data/indexes/bm25_index.json \
  --question "mobile segmentation limits" --top-k 8
```

### `vector-service-health`

Check vector sidecar readiness and metadata.

```bash
python -m src.cli vector-service-health --index data/indexes/bm25_index.json
```

---

## Corpus & Extraction

### `extract-corpus`

Extract text from downloaded PDFs.

```bash
python -m src.cli extract-corpus --papers-dir data/papers --out-dir data/extracted \
  --db-path data/papers.db --layout-engine shadow
```

| Argument            | Default          | Description                         |
| ------------------- | ---------------- | ----------------------------------- |
| `--papers-dir`      | `data/papers`    | PDF directory                       |
| `--out-dir`         | `data/extracted` | Output directory                    |
| `--layout-engine`   | `shadow`         | `legacy`, `v2`, `shadow`, or `auto` |
| `--two-column-mode` | `auto`           | `off`, `auto`, or `force`           |
| `--ocr-enabled`     | config/false     | Enable OCR fallback                 |
| `--ocr-lang`        | `eng`            | Tesseract language (or `auto`)      |

### `corpus-health`

Evaluate extracted corpus quality before indexing.

```bash
python -m src.cli corpus-health --corpus data/extracted \
  --min-snippets 1 --min-avg-chars-per-paper 500
```

### `extraction-quality-report`

Generate per-PDF extraction quality report.

```bash
python -m src.cli extraction-quality-report --corpus data/extracted \
  --out runs/research_reports/extraction_quality.md
```

### `extraction-eval`

Evaluate extraction against gold-standard checks.

```bash
python -m src.cli extraction-eval --corpus data/extracted \
  --gold tests/fixtures/extraction_gold.json --fail-below 0.75
```

### `layout-promotion-gate`

Evaluate whether layout v2 can be promoted from shadow mode.

```bash
python -m src.cli layout-promotion-gate --papers-dir data/papers \
  --gold tests/fixtures/extraction_gold.json
```

### `migrate-extraction-meta`

Backfill extraction metadata for legacy corpus JSON files.

```bash
python -m src.cli migrate-extraction-meta --corpus data/extracted --dry-run
```

---

## Utility

### `snapshot-run`

Create a reproducibility snapshot bundle.

```bash
python -m src.cli snapshot-run --report runs/research_reports/report.md \
  --index data/indexes/bm25_index.json --config config/train.yaml \
  --config config/sources.yaml --out runs/snapshots
```

### `doc-write`

Generate documentation from structured facts.

```bash
python -m src.cli doc-write --facts project_facts.json --doc-type readme
```

### `release-notes`

Generate release notes between two git refs.

```bash
python -m src.cli release-notes --from v0.1.0 --to v0.2.0
```

### `scaffold-domain-config`

Create a new domain sources config template.

```bash
python -m src.cli scaffold-domain-config --domain "quantum computing" \
  --out config/domains/sources_quantum_computing.yaml
```
