import json
from pathlib import Path

from src.apps.snapshot import create_snapshot


def test_create_snapshot_bundle(tmp_path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    report.with_suffix(".json").write_text('{"question":"q","shortlist":[],"synthesis":"Not found in sources.","gaps":[],"experiments":[],"citations":[]}', encoding="utf-8")
    report.with_suffix(".evidence.json").write_text('{"question":"q","items":[]}', encoding="utf-8")
    report.with_suffix(".metrics.json").write_text('{"cache_hit":false}', encoding="utf-8")

    out = create_snapshot(str(tmp_path / "snaps"), report_path=str(report), config_paths=["config/train.yaml"])
    root = Path(out)
    assert root.exists()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "python" in manifest
    assert manifest["artifacts"]
