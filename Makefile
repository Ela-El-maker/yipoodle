PY := .venv/bin/python

QUERY ?= mobile segmentation limitations
TOP_K ?= 8
MIN_ITEMS ?= 2
MIN_SCORE ?= 0.5
REPORT ?= runs/research_reports/report.md
INDEX ?= data/indexes/bm25_index.json
CORPUS ?= data/extracted
DB ?= data/papers.db
PAPERS_DIR ?= data/papers
MIN_SNIPPETS ?= 1
MIN_AVG_CHARS_PER_PAPER ?= 500
MIN_AVG_CHARS_PER_PAGE ?= 80
MAX_EXTRACT_ERROR_RATE ?= 0.8
SOURCES_CONFIG ?= config/sources.yaml
EXTRACTION_GOLD ?= tests/fixtures/extraction_gold.json
EXTRACTION_EVAL_OUT ?= runs/research_reports/extraction_eval.md
EXTRACTION_EVAL_CORPUS ?= tests/fixtures/extracted

.PHONY: test lint setup-env report-local report pipeline corpus-health extraction-eval extraction-eval-live benchmark benchmark-scale benchmark-regression-check reliability-watchdog research-template watch-ingest-once snapshot auto-update docker-build-cpu docker-build-cuda docker-compose-cpu-up docker-compose-cpu-down

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m pip install -q ruff
	$(PY) -m ruff check src tests --select E9,F63,F7,F82

setup-env:
	bash scripts/bootstrap_env.sh --profile cpu

report-local:
	$(PY) -m src.cli build-index --corpus $(CORPUS) --out $(INDEX) --db-path $(DB) --require-healthy-corpus --min-snippets $(MIN_SNIPPETS) --min-avg-chars-per-paper $(MIN_AVG_CHARS_PER_PAPER) --min-avg-chars-per-page $(MIN_AVG_CHARS_PER_PAGE) --max-extract-error-rate $(MAX_EXTRACT_ERROR_RATE)
	$(PY) -m src.cli research --index $(INDEX) --question "$(QUERY)" --top-k $(TOP_K) --min-items $(MIN_ITEMS) --min-score $(MIN_SCORE) --sources-config $(SOURCES_CONFIG) --out $(REPORT)
	$(PY) -m src.cli validate-report --input $(REPORT) --evidence $(basename $(REPORT)).evidence.json

corpus-health:
	$(PY) -m src.cli corpus-health --corpus $(CORPUS) --min-snippets $(MIN_SNIPPETS) --min-avg-chars-per-paper $(MIN_AVG_CHARS_PER_PAPER) --min-avg-chars-per-page $(MIN_AVG_CHARS_PER_PAGE) --max-extract-error-rate $(MAX_EXTRACT_ERROR_RATE)

extraction-eval:
	$(PY) -m src.cli extraction-eval --corpus $(EXTRACTION_EVAL_CORPUS) --gold $(EXTRACTION_GOLD) --out $(EXTRACTION_EVAL_OUT)

extraction-eval-live:
	$(PY) -m src.cli extraction-eval --corpus $(CORPUS) --gold $(EXTRACTION_GOLD) --out $(EXTRACTION_EVAL_OUT)

report:
	$(PY) -m src.cli sync-papers --query "$(QUERY)" --max-results 20 --db-path $(DB) --papers-dir $(PAPERS_DIR) --with-semantic-scholar --sources-config $(SOURCES_CONFIG)
	$(PY) -m src.cli extract-corpus --papers-dir $(PAPERS_DIR) --out-dir $(CORPUS) --db-path $(DB)
	$(MAKE) report-local QUERY="$(QUERY)" TOP_K=$(TOP_K) MIN_ITEMS=$(MIN_ITEMS) MIN_SCORE=$(MIN_SCORE) REPORT=$(REPORT) INDEX=$(INDEX) CORPUS=$(CORPUS)

pipeline:
	$(PY) -m src.cli run-pipeline --query "$(QUERY)" --max-results 20 --db-path $(DB) --papers-dir $(PAPERS_DIR) --extracted-dir $(CORPUS) --index $(INDEX) --report $(REPORT) --with-semantic-scholar --top-k $(TOP_K) --min-items $(MIN_ITEMS) --min-score $(MIN_SCORE) --sources-config $(SOURCES_CONFIG)

benchmark:
	$(PY) -m src.cli benchmark-research --index $(INDEX) --queries-file tests/fixtures/queries.txt --runs-per-query 3 --sources-config $(SOURCES_CONFIG) --out runs/research_reports/benchmark.json

benchmark-scale:
	$(PY) -m src.cli benchmark-scale --corpus $(CORPUS) --queries-file tests/fixtures/queries.txt --repeat-factor 50 --runs-per-query 2 --out runs/research_reports/benchmark_scale.json

benchmark-regression-check:
	$(PY) -m src.cli benchmark-regression-check --benchmark runs/research_reports/benchmark.json --history runs/audit/benchmark_history.json --max-latency-regression-pct 10 --min-quality-floor 0.0

reliability-watchdog:
	$(PY) -m src.cli reliability-watchdog --run-dir runs/audit/runs/$$(ls -1 runs/audit/runs | tail -n1) --config config/automation.yaml

research-template:
	$(PY) -m src.cli research-template --template lit_review --topic "$(QUERY)" --index $(INDEX) --templates-config config/templates.yaml

watch-ingest-once:
	$(PY) -m src.cli watch-ingest --dir $(PAPERS_DIR) --once --db-path $(DB) --extracted-dir $(CORPUS) --index $(INDEX)

snapshot:
	$(PY) -m src.cli snapshot-run --report $(REPORT) --index $(INDEX) --config config/train.yaml --config config/sources.yaml --out runs/snapshots

auto-update:
	bash scripts/auto_update.sh

docker-build-cpu:
	docker build -f docker/Dockerfile.cpu -t yipoodle:cpu .

docker-build-cuda:
	docker build -f docker/Dockerfile.cuda -t yipoodle:cuda .

docker-compose-cpu-up:
	docker compose -f deploy/docker-compose.cpu.yml up -d --build

docker-compose-cpu-down:
	docker compose -f deploy/docker-compose.cpu.yml down
