from src.apps.live_routing import IntentResult, build_not_found_diagnostics, detect_intent, sources_for_intent


def test_detect_intent_weather_terms() -> None:
    cfg = {
        "live_routing": {
            "intents": {
                "weather_now": {"match_terms": ["weather", "temperature"], "sources": ["open_meteo"]}
            }
        }
    }
    out = detect_intent("What is the weather and temperature today?", cfg)
    assert out.intent == "weather_now"
    assert out.confidence > 0
    assert "weather" in out.matched_terms


def test_sources_for_intent_filters_to_enabled() -> None:
    cfg = {
        "live_routing": {
            "intents": {
                "finance_price": {"match_terms": ["price"], "sources": ["coingecko_price", "yahoo_finance"]}
            }
        }
    }
    enabled = {"coingecko_price": object()}
    out = sources_for_intent("finance_price", cfg, enabled)
    assert out == ["coingecko_price"]


def test_missing_intent_sources_returns_empty() -> None:
    cfg = {
        "live_routing": {
            "intents": {
                "sports_results": {"match_terms": ["score"], "sources": ["sports_results_api"]}
            }
        }
    }
    enabled: dict[str, object] = {}
    out = sources_for_intent("sports_results", cfg, enabled)
    assert out == []


def test_build_not_found_diagnostics_payload() -> None:
    cfg = {"live_routing_help": {"weather_now": {"suggest_enable": ["open_meteo"]}}}
    d = build_not_found_diagnostics(
        question="What is weather?",
        intent_result=IntentResult(intent="weather_now", confidence=0.9, matched_terms=["weather"]),
        attempted_sources=[],
        reject_reason="no_sources_for_intent",
        cfg=cfg,
    )
    assert d["intent_detected"] == "weather_now"
    assert d["reject_reason"] == "no_sources_for_intent"
    assert d["suggest_enable"] == ["open_meteo"]
