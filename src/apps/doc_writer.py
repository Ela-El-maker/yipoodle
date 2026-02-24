from __future__ import annotations

from pathlib import Path
import yaml


def _render_readme(facts: dict) -> str:
    title = facts.get("project_name", "Project")
    summary = facts.get("summary", "")
    stack = facts.get("stack", [])
    commands = facts.get("commands", [])
    lines = [f"# {title}", "", summary, "", "## Stack"]
    lines.extend([f"- {s}" for s in stack])
    lines.append("\n## Commands")
    lines.extend([f"```bash\n{c}\n```" for c in commands])
    return "\n".join(lines) + "\n"


def _render_arch(facts: dict) -> str:
    lines = ["# Architecture", "", "## Components"]
    for c in facts.get("components", []):
        lines.append(f"- {c}")
    lines.append("\n## Data Flow")
    for s in facts.get("data_flow", []):
        lines.append(f"- {s}")
    return "\n".join(lines) + "\n"


def _render_api(facts: dict) -> str:
    lines = ["# API", ""]
    for ep in facts.get("endpoints", []):
        lines.append(f"## {ep.get('method', 'GET')} {ep.get('path', '/')}")
        lines.append(ep.get("description", ""))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_doc(doc_type: str, facts_path: str, out_path: str | None = None) -> str:
    facts = yaml.safe_load(Path(facts_path).read_text(encoding="utf-8"))
    if doc_type == "readme":
        content = _render_readme(facts)
        out = out_path or "README.generated.md"
    elif doc_type == "arch":
        content = _render_arch(facts)
        out = out_path or "ARCHITECTURE.generated.md"
    elif doc_type == "api":
        content = _render_api(facts)
        out = out_path or "API.generated.md"
    else:
        raise ValueError("doc_type must be one of: readme|arch|api")

    Path(out).write_text(content, encoding="utf-8")
    return out
