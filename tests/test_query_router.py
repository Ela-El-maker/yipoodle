from src.apps.query_router import dispatch_query, load_router_config, route_query


def test_route_query_explicit_override_wins() -> None:
    cfg = load_router_config(None)
    out = route_query("monitor nvidia", cfg, explicit_mode="ask")
    assert out.mode == "ask"
    assert out.override_used is True
    assert out.reason == "explicit_mode_override"


def test_route_query_monitor_keywords() -> None:
    cfg = load_router_config(None)
    out = route_query("Monitor NVIDIA stock and alert me", cfg)
    assert out.mode == "monitor"
    assert out.override_used is False
    assert out.reason == "monitor_keywords"


def test_route_query_notes_keywords() -> None:
    cfg = load_router_config(None)
    out = route_query("Create study notes on transformers", cfg)
    assert out.mode == "notes"
    assert out.reason == "notes_keywords"


def test_route_query_math_to_ask() -> None:
    cfg = load_router_config(None)
    out = route_query("23 + 34 = ?", cfg)
    assert out.mode == "ask"
    assert out.reason == "direct_arithmetic_candidate"


def test_route_query_short_definition_to_ask() -> None:
    cfg = load_router_config(None)
    out = route_query("What is an algorithm?", cfg)
    assert out.mode == "ask"


def test_route_query_definition_keyword_anywhere_to_ask() -> None:
    cfg = load_router_config(None)
    out = route_query("Can you explain amortization in simple terms?", cfg)
    assert out.mode == "ask"


def test_route_query_factoid_who_is_to_ask() -> None:
    cfg = load_router_config(None)
    out = route_query("Who is Barack Obama?", cfg)
    assert out.mode == "ask"


def test_route_query_ambiguous_defaults_research() -> None:
    cfg = load_router_config(None)
    out = route_query("Compare optimization techniques for sparse transformers in long-context forecasting", cfg)
    assert out.mode == "research"
    assert out.reason == "default_research_fallback"


def test_dispatch_query_calls_selected_handler() -> None:
    cfg = load_router_config(None)
    decision = route_query("23 + 34 = ?", cfg)
    called = {"ask": False}

    def _ask():
        called["ask"] = True
        return {"ok": True}

    out = dispatch_query(decision, {"ask": _ask})
    assert called["ask"] is True
    assert out == {"ok": True}
