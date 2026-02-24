#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.apps.automation import dispatch_alerts, load_automation_config, write_summary_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latest automation run summary")
    parser.add_argument("--audit-dir", default="runs/audit")
    parser.add_argument("--run-dir", default=None, help="Optional explicit run directory")
    parser.add_argument("--config", default="config/automation.yaml")
    parser.add_argument("--out-json", default="runs/audit/latest_summary.json")
    parser.add_argument("--out-md", default="runs/audit/latest_summary.md")
    parser.add_argument("--alerts-out-json", default="runs/audit/latest_alert.json")
    args = parser.parse_args()

    try:
        summary = write_summary_outputs(
            audit_dir=args.audit_dir,
            run_dir=args.run_dir,
            out_json=args.out_json,
            out_md=args.out_md,
        )
        cfg = load_automation_config(args.config)
        alert_result = dispatch_alerts(summary, cfg)
        Path(args.alerts_out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.alerts_out_json).write_text(json.dumps(alert_result, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(json.dumps({"alert": alert_result}, indent=2))
    except FileNotFoundError as exc:
        print(json.dumps({"status": "no_summary_available", "reason": str(exc)}, indent=2))


if __name__ == "__main__":
    main()
