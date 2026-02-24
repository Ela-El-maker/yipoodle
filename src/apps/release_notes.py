from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import subprocess


def _bucket(commits: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for c in commits:
        lower = c.lower()
        if "breaking" in lower:
            buckets["Breaking"].append(c)
        elif lower.startswith("feat"):
            buckets["Added"].append(c)
        elif lower.startswith("fix"):
            buckets["Fixed"].append(c)
        elif lower.startswith("perf"):
            buckets["Performance"].append(c)
        else:
            buckets["Changed"].append(c)
    return buckets


def _read_commits(from_ref: str, to_ref: str) -> list[str]:
    cmd = ["git", "log", "--pretty=%s", f"{from_ref}..{to_ref}"]
    out = subprocess.check_output(cmd, text=True).strip()
    return [line for line in out.splitlines() if line]


def generate_release_notes(from_ref: str, to_ref: str, out_path: str) -> str:
    commits = _read_commits(from_ref, to_ref)
    buckets = _bucket(commits)

    lines = [f"# Release Notes {to_ref}", ""]
    for section in ["Added", "Fixed", "Changed", "Performance", "Breaking"]:
        items = buckets.get(section, [])
        if not items:
            continue
        lines.append(f"## {section}")
        lines.extend([f"- {i}" for i in items])
        lines.append("")

    if not commits:
        lines.append("No changes in this range.")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
