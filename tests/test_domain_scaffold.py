from pathlib import Path

import yaml

from src.apps.domain_scaffold import scaffold_domain_config


def test_scaffold_domain_config_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "sources_marketing.yaml"
    path = scaffold_domain_config(domain="Marketing Growth", out_path=str(out), profile="auto", overwrite=False)
    assert path == str(out)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert payload["domain"] == "marketing_growth"
    assert payload["sources"]["openalex"]["enabled"] is True
    assert payload["sources"]["crossref"]["enabled"] is True
    assert payload["ocr"]["enabled"] is False
    assert payload["ocr"]["lang"] == "eng"


def test_scaffold_domain_config_respects_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "sources_x.yaml"
    out.write_text("domain: x\n", encoding="utf-8")
    try:
        scaffold_domain_config(domain="x", out_path=str(out), overwrite=False)
        assert False, "Expected FileExistsError"
    except FileExistsError:
        pass
