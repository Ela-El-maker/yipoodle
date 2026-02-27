from __future__ import annotations

from src.cli import build_parser


def test_parser_has_serve_ui_command() -> None:
    p = build_parser()
    args = p.parse_args(["serve-ui", "--config", "config/ui.yaml", "--host", "127.0.0.1", "--port", "8080"])
    assert args.command == "serve-ui"
    assert args.config == "config/ui.yaml"
    assert args.host == "127.0.0.1"
    assert args.port == 8080
