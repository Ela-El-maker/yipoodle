#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="cpu"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WITH_DOCKER_BUILD=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_env.sh [options]

Options:
  --profile <base|cpu|cuda12|dev>  Dependency profile to install (default: cpu)
  --python <path>                  Python binary to use for venv (default: python3)
  --with-docker-build              Build Docker CPU image after setup
  -h, --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --with-docker-build)
      WITH_DOCKER_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

case "$PROFILE" in
  base)
    pip install -r deploy/requirements.base.txt
    ;;
  cpu)
    pip install -r deploy/requirements.cpu.txt
    ;;
  cuda12)
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r deploy/requirements.cuda12.txt
    ;;
  dev)
    pip install -r requirements.txt
    ;;
  *)
    echo "Unsupported profile: $PROFILE" >&2
    exit 1
    ;;
esac

mkdir -p \
  data/papers \
  data/extracted \
  data/indexes \
  data/kb \
  data/reliability \
  runs/audit \
  runs/research_reports \
  runs/monitor \
  runs/notes \
  runs/sessions \
  runs/live

if [[ ! -f .env.example ]]; then
  cat > .env.example <<'EOF'
UNPAYWALL_EMAIL=
GMAIL_APP_PASSWORD=
OPENAI_API_KEY=
EOF
fi

if [[ "$WITH_DOCKER_BUILD" -eq 1 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed; skipping image build." >&2
  else
    docker build -f docker/Dockerfile.cpu -t yipoodle:cpu .
  fi
fi

cat <<EOF
Bootstrap complete.
Profile: $PROFILE
Python: $(python --version 2>/dev/null || true)
Virtual environment: $ROOT_DIR/.venv

Next steps:
  source .venv/bin/activate
  python -m src.cli --help
  pytest -q
EOF
