import json

from src.apps.ask_mode import answer_ask_question, run_ask_mode
from src.apps.query_router import load_router_config


def test_ask_mode_arithmetic_handler() -> None:
    cfg = load_router_config(None)
    md, meta = answer_ask_question(question="23 + 34 = ?", router_cfg=cfg)
    assert "57" in md
    assert meta["ask_handler_used"] == "direct_arithmetic"
    assert meta["deterministic"] is True


def test_ask_mode_unit_conversion_handler() -> None:
    cfg = load_router_config(None)
    md, meta = answer_ask_question(question="45 km/h to m/s", router_cfg=cfg)
    assert "12.5 m/s" in md
    assert meta["ask_handler_used"] == "unit_conversion"


def test_ask_mode_glossary_lookup(tmp_path) -> None:
    gloss = tmp_path / "ask_glossary.yaml"
    gloss.write_text("glossary:\n  algorithm: test definition\n", encoding="utf-8")
    cfg = load_router_config(None)
    md, meta = answer_ask_question(question="What is algorithm?", router_cfg=cfg, glossary_path=str(gloss))
    assert "test definition" in md
    assert meta["ask_handler_used"] == "glossary"


def test_ask_mode_glossary_lookup_with_qualifier(tmp_path) -> None:
    gloss = tmp_path / "ask_glossary.yaml"
    gloss.write_text("glossary:\n  overfitting: test overfitting definition\n", encoding="utf-8")
    cfg = load_router_config(None)
    md, meta = answer_ask_question(
        question="Can you explain overfitting in simple terms?",
        router_cfg=cfg,
        glossary_path=str(gloss),
    )
    assert "test overfitting definition" in md
    assert meta["ask_handler_used"] == "glossary"


def test_ask_mode_citation_notice_stays_ask() -> None:
    cfg = load_router_config(None)
    md, meta = answer_ask_question(question="What is an algorithm with citations?", router_cfg=cfg)
    assert "ASK mode does not provide evidence citations" in md
    assert meta["no_citation_notice_emitted"] is True


def test_ask_mode_fallback() -> None:
    cfg = load_router_config(None)
    md, meta = answer_ask_question(
        question="global payment rails reshape macro liquidity under policy shocks",
        router_cfg=cfg,
        glossary_path=None,
    )
    assert "Use RESEARCH for grounded output" in md
    assert meta["ask_handler_used"] == "fallback"


def test_ask_mode_definition_fallback() -> None:
    cfg = load_router_config(None)
    md, meta = answer_ask_question(
        question="Explain amortization schedule mechanics in simple terms",
        router_cfg=cfg,
        glossary_path=None,
    )
    assert "don't have a reliable local ASK definition" in md
    assert meta["ask_handler_used"] == "definition_fallback"


def test_run_ask_mode_writes_sidecar(tmp_path) -> None:
    cfg = load_router_config(None)
    out = tmp_path / "ask.md"
    payload = run_ask_mode(question="23+34", out_path=str(out), router_cfg=cfg, glossary_path=None)
    sidecar = out.with_suffix(".json")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["out_path"] == str(out)
    assert data["mode"] == "ask"
    assert data["ask_handler_used"] == "direct_arithmetic"
