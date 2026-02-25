# Deployment

This document covers deployment options for Yipoodle, including Docker containers, production dependency profiles, and cron-based automation setup.

---

## Deployment Options

| Method          | Best For                               |
| --------------- | -------------------------------------- |
| Local venv      | Development, single-user research      |
| Docker CPU      | Portable deployment, CI/CD pipelines   |
| Docker CUDA     | GPU-accelerated training and embedding |
| Cron automation | Scheduled unattended research runs     |

---

## CI/CD and Repository Triggers

GitHub Actions workflow: `.github/workflows/ci.yml`

- `pull_request`: runs lint, pytest, and quality gates.
- `push` to `main` and version tags: runs the same checks, then builds/pushes CPU image to GHCR.
- `workflow_dispatch`: manual trigger support.

This provides automated validation on PRs and automated build/deploy on push.

---

## Local Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli --help
```

### Optional Extras

```bash
# Better PDF extraction
pip install pymupdf pdfminer.six

# OCR support
sudo apt-get install tesseract-ocr
```

---

## Production Dependency Profiles

Pinned production profiles are provided under `deploy/`:

| Profile           | File                             | Notes                                  |
| ----------------- | -------------------------------- | -------------------------------------- |
| Base (no PyTorch) | `deploy/requirements.base.txt`   | Core dependencies without ML framework |
| CPU               | `deploy/requirements.cpu.txt`    | Includes PyTorch CPU + FAISS CPU       |
| CUDA 12.1         | `deploy/requirements.cuda12.txt` | PyTorch with CUDA support              |

### CPU Install

```bash
pip install -r deploy/requirements.cpu.txt
```

### CUDA 12.1 Install

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r deploy/requirements.cuda12.txt
```

---

## Docker

### Build Images

```bash
# CPU image
make docker-build-cpu
# or: docker build -f docker/Dockerfile.cpu -t yipoodle:cpu .

# CUDA image
make docker-build-cuda
# or: docker build -f docker/Dockerfile.cuda -t yipoodle:cuda .
```

### Image Details

**CPU Image** (`docker/Dockerfile.cpu`):

- Base: `python:3.11-slim`
- Installs `build-essential` and `curl`
- Uses `deploy/requirements.cpu.txt`
- Default CMD: `python -m src.cli --help`

**CUDA Image** (`docker/Dockerfile.cuda`):

- Base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- Uses `deploy/requirements.base.txt` + faiss-cpu
- Default CMD: `python -m src.cli --help`

### Running Containers

```bash
# Run a research pipeline
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs \
  yipoodle:cpu python -m src.cli run-pipeline \
  --query "transformer architectures" --sources-config config/sources.yaml

# Interactive shell
docker run --rm -it -v $(pwd)/data:/app/data yipoodle:cpu bash
```

### Automated Bootstrap

```bash
# Local venv + dependencies + standard directory provisioning
bash scripts/bootstrap_env.sh --profile cpu

# Optional: include Docker image build
bash scripts/bootstrap_env.sh --profile cpu --with-docker-build
```

### Compose Provisioning (CPU)

```bash
docker compose -f deploy/docker-compose.cpu.yml up -d --build
docker compose -f deploy/docker-compose.cpu.yml down
```

### Volume Mounts

Mount these directories to persist data across container runs:

| Host Path  | Container Path | Purpose                             |
| ---------- | -------------- | ----------------------------------- |
| `./data`   | `/app/data`    | Papers, extracted text, indexes, KB |
| `./runs`   | `/app/runs`    | Reports, checkpoints, audit trails  |
| `./config` | `/app/config`  | Custom configuration overrides      |

---

## Cron-Based Automation

Yipoodle includes a full automation engine for scheduled research runs.

### How It Works

1. `scripts/auto_update.sh` is the cron entry point.
2. It acquires a filesystem lock (prevents concurrent runs).
3. Calls `scripts/auto_update.py` which runs `run_automation()` from `src/apps/automation.py`.
4. After the run, `scripts/post_run_summary.py` generates summary artifacts and dispatches alerts.

### Setup

#### 1. Configure Topics

Edit `config/automation.yaml` to define research topics:

```yaml
topics:
  mobile_segmentation:
    query: "mobile real-time semantic segmentation edge deployment"
    max_results: 20
    top_k: 12
    min_items: 4
    min_score: 0.15
```

#### 2. Test Locally

```bash
make auto-update
```

This runs the full pipeline once and writes results to `runs/audit/`.

#### 3. Install Cron Job

```bash
crontab -e
```

Add a weekly schedule (Sunday 2AM):

```cron
0 2 * * 0 /path/to/yipoodle/scripts/auto_update.sh >> /path/to/yipoodle/runs/audit/cron.log 2>&1
```

#### 4. Remove Cron Job

```bash
crontab -e
# Delete the auto_update line
```

### Automation Outputs

| Artifact              | Path                                         | Description               |
| --------------------- | -------------------------------------------- | ------------------------- |
| Run manifest          | `runs/audit/runs/<run_id>/manifest.json`     | Per-run detailed manifest |
| Latest summary (JSON) | `runs/audit/latest_summary.json`             | Machine-readable summary  |
| Latest summary (MD)   | `runs/audit/latest_summary.md`               | Human-readable summary    |
| Alert dispatch        | `runs/audit/latest_alert.json`               | Alert delivery status     |
| Topic reports         | `runs/research_reports/automation/<run_id>/` | Per-topic reports         |
| Cron log              | `runs/audit/cron.log`                        | Shell output log          |

### Alerting

Configure alerts in `config/automation.yaml`:

```yaml
alerts:
  enabled: true
  # Webhook (Slack, Teams, etc.)
  webhook_url: "https://hooks.slack.com/services/..."
  webhook_timeout_sec: 10
  webhook_headers:
    Authorization: "Bearer <token>"
  # Email (Gmail SMTP)
  email_enabled: true
  email_to: ["team@example.com"]
  email_from: "alerts@example.com"
  smtp_host: "smtp.gmail.com"
  smtp_port: 465
  smtp_use_ssl: true
  smtp_username: "alerts@example.com"
  smtp_password_env: "GMAIL_APP_PASSWORD"
  # Alert triggers
  on_corpus_unhealthy: true
  on_topic_validation_failed: true
  on_source_errors: true
```

Gmail authentication requires an App Password:

```bash
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
```

### Replay and Recovery

Replay the latest summary with alert dispatch:

```bash
.venv/bin/python scripts/post_run_summary.py --audit-dir runs/audit --config config/automation.yaml
```

Manually rerun a failed topic:

```bash
python -m src.cli research --index data/indexes/bm25_index.json \
  --question "<topic query>" --top-k 12 --min-items 4 --min-score 0.15 \
  --out runs/research_reports/replay.md

python -m src.cli validate-report --input runs/research_reports/replay.md \
  --evidence runs/research_reports/replay.evidence.json
```

---

## Environment Variables

| Variable             | Used By           | Required | Description                       |
| -------------------- | ----------------- | -------- | --------------------------------- |
| `UNPAYWALL_EMAIL`    | `sync-papers`     | No       | Email for Unpaywall OA PDF lookup |
| `GMAIL_APP_PASSWORD` | Automation alerts | No       | Gmail App Password for SMTP       |
| `OPENAI_API_KEY`     | `validate-report` | No       | API key for online semantic judge |

---

## Resource Requirements

### Minimum (CPU-only, small corpus)

- Python 3.11+
- 2 GB RAM
- 1 GB disk (plus corpus storage)

### Recommended (hybrid retrieval, medium corpus)

- 8 GB RAM (sentence-transformers loading)
- 4+ CPU cores (parallel extraction benefits)
- 10 GB disk

### GPU (training / large-scale embeddings)

- NVIDIA GPU with CUDA 12.1
- 16 GB+ RAM
- PyTorch CUDA build
