#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.apps.automation import run_automation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cron-style automation pipeline")
    parser.add_argument("--config", default="config/automation.yaml")
    args = parser.parse_args()
    run_dir = run_automation(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
