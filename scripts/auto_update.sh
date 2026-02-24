#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"
CONFIG="$ROOT_DIR/config/automation.yaml"
AUDIT_DIR="$ROOT_DIR/runs/audit"
LOCK_DIR="$AUDIT_DIR/.auto_update.lock"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$AUDIT_DIR/auto_update_${STAMP}.log"

mkdir -p "$AUDIT_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "auto_update lock exists: $LOCK_DIR"
  exit 99
fi

cleanup() {
  rmdir "$LOCK_DIR" || true
}
trap cleanup EXIT

exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== AUTO UPDATE START ==="
echo "utc=$STAMP"
echo "root=$ROOT_DIR"

status=0
"$PY" "$ROOT_DIR/scripts/auto_update.py" --config "$CONFIG" || status=$?
"$PY" "$ROOT_DIR/scripts/post_run_summary.py" --audit-dir "$AUDIT_DIR" \
  --config "$CONFIG" \
  --out-json "$AUDIT_DIR/latest_summary.json" \
  --out-md "$AUDIT_DIR/latest_summary.md" \
  --alerts-out-json "$AUDIT_DIR/latest_alert.json" || true

echo "=== AUTO UPDATE END === status=$status"
exit "$status"
