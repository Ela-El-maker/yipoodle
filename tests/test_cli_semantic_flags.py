from src.cli import build_parser


def test_validate_report_semantic_flag_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["validate-report", "--input", "x.md", "--evidence", "x.evidence.json"])
    assert args.semantic_faithfulness is True
    assert args.semantic_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert args.semantic_min_support == 0.55
    assert args.semantic_max_contradiction == 0.30
    assert args.semantic_shadow_mode is True
    assert args.semantic_fail_on_low_support is False


def test_validate_report_semantic_flag_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "validate-report",
            "--input",
            "x.md",
            "--evidence",
            "x.evidence.json",
            "--no-semantic-faithfulness",
            "--semantic-model",
            "sentence-transformers/all-mpnet-base-v2",
            "--semantic-min-support",
            "0.6",
            "--semantic-max-contradiction",
            "0.2",
            "--no-semantic-shadow-mode",
            "--semantic-fail-on-low-support",
        ]
    )
    assert args.semantic_faithfulness is False
    assert args.semantic_model == "sentence-transformers/all-mpnet-base-v2"
    assert args.semantic_min_support == 0.6
    assert args.semantic_max_contradiction == 0.2
    assert args.semantic_shadow_mode is False
    assert args.semantic_fail_on_low_support is True

