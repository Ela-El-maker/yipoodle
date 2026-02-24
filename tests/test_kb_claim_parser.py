from src.apps.kb_claim_parser import parse_key_claims


def test_parse_key_claims_extracts_bullets_with_citations() -> None:
    md = """
# Research Report

## Key Claims
- Transformer models can fail under distribution shift. (Pabc:S1)
- Live price signals are noisy intraday. (SNAP:abc123:S2)

## Gaps
- more data
"""
    rows = parse_key_claims(md)
    assert len(rows) == 2
    assert rows[0].claim_text.startswith("Transformer models")
    assert rows[0].citations == ["(Pabc:S1)"]
    assert rows[1].citations == ["(SNAP:abc123:S2)"]


def test_parse_key_claims_missing_section_returns_empty() -> None:
    assert parse_key_claims("## Synthesis\n- x") == []
