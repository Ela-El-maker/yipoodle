from __future__ import annotations

from pathlib import Path

import yaml

import src.cli as cli


def _write_sources(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_extract_corpus_uses_sources_ocr_defaults(monkeypatch, tmp_path) -> None:
    sources_path = tmp_path / "sources.yaml"
    _write_sources(
        sources_path,
        {
            "domain": "finance_markets",
            "ocr": {
                "enabled": True,
                "timeout_sec": 44,
                "min_chars_trigger": 101,
                "max_pages": 11,
                "min_output_chars": 222,
                "min_gain_chars": 55,
                "min_confidence": 60.5,
                "lang": "eng+deu",
                "profile": "sparse",
                "noise_suppression": False,
            },
        },
    )

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--papers-dir",
            "data/papers",
            "--out-dir",
            "data/extracted",
            "--sources-config",
            str(sources_path),
        ]
    )
    monkeypatch.setattr(cli.sys, "argv", ["prog", "extract-corpus", "--sources-config", str(sources_path)])
    captured: dict = {}

    def _fake_extract(*_a, **kwargs):
        captured.update(kwargs)
        return {"ok": 1}

    monkeypatch.setattr(cli, "extract_from_papers_dir", _fake_extract)
    cli.cmd_extract_corpus(args)

    assert captured["ocr_enabled"] is True
    assert captured["ocr_timeout_sec"] == 44
    assert captured["ocr_min_chars_trigger"] == 101
    assert captured["ocr_max_pages"] == 11
    assert captured["ocr_min_output_chars"] == 222
    assert captured["ocr_min_gain_chars"] == 55
    assert captured["ocr_min_confidence"] == 60.5
    assert captured["ocr_lang"] == "eng+deu"
    assert captured["ocr_profile"] == "sparse"
    assert captured["ocr_noise_suppression"] is False


def test_extract_corpus_cli_overrides_sources_ocr(monkeypatch, tmp_path) -> None:
    sources_path = tmp_path / "sources.yaml"
    _write_sources(
        sources_path,
        {
            "domain": "finance_markets",
            "ocr": {
                "enabled": False,
                "lang": "eng",
                "profile": "document",
                "min_confidence": 40.0,
                "noise_suppression": True,
            },
        },
    )
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--papers-dir",
            "data/papers",
            "--out-dir",
            "data/extracted",
            "--sources-config",
            str(sources_path),
            "--ocr-enabled",
            "--ocr-lang",
            "eng+spa",
            "--ocr-profile",
            "sparse",
            "--ocr-min-confidence",
            "77.0",
            "--no-ocr-noise-suppression",
        ]
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "prog",
            "extract-corpus",
            "--sources-config",
            str(sources_path),
            "--ocr-enabled",
            "--ocr-lang",
            "eng+spa",
            "--ocr-profile",
            "sparse",
            "--ocr-min-confidence",
            "77.0",
            "--no-ocr-noise-suppression",
        ],
    )
    captured: dict = {}

    def _fake_extract(*_a, **kwargs):
        captured.update(kwargs)
        return {"ok": 1}

    monkeypatch.setattr(cli, "extract_from_papers_dir", _fake_extract)
    cli.cmd_extract_corpus(args)

    assert captured["ocr_enabled"] is True
    assert captured["ocr_lang"] == "eng+spa"
    assert captured["ocr_profile"] == "sparse"
    assert captured["ocr_min_confidence"] == 77.0
    assert captured["ocr_noise_suppression"] is False


def test_extract_corpus_falls_back_when_ocr_block_missing(monkeypatch, tmp_path) -> None:
    sources_path = tmp_path / "sources.yaml"
    _write_sources(sources_path, {"domain": "custom"})
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "extract-corpus",
            "--papers-dir",
            "data/papers",
            "--out-dir",
            "data/extracted",
            "--sources-config",
            str(sources_path),
        ]
    )
    monkeypatch.setattr(cli.sys, "argv", ["prog", "extract-corpus", "--sources-config", str(sources_path)])
    captured: dict = {}

    def _fake_extract(*_a, **kwargs):
        captured.update(kwargs)
        return {"ok": 1}

    monkeypatch.setattr(cli, "extract_from_papers_dir", _fake_extract)
    cli.cmd_extract_corpus(args)

    assert captured["ocr_enabled"] is False
    assert captured["ocr_timeout_sec"] == 30
    assert captured["ocr_min_chars_trigger"] == 120
    assert captured["ocr_max_pages"] == 20
    assert captured["ocr_min_output_chars"] == 200
    assert captured["ocr_min_gain_chars"] == 40
    assert captured["ocr_min_confidence"] == 45.0
    assert captured["ocr_lang"] == "eng"
    assert captured["ocr_profile"] == "document"
    assert captured["ocr_noise_suppression"] is True
