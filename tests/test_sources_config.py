from src.apps.sources_config import ocr_config


def test_ocr_config_missing_block_returns_empty() -> None:
    assert ocr_config({}) == {}


def test_ocr_config_normalizes_values() -> None:
    cfg = {
        "ocr": {
            "enabled": True,
            "timeout_sec": "45",
            "min_chars_trigger": "110",
            "max_pages": 15,
            "min_output_chars": 250,
            "min_gain_chars": 60,
            "min_confidence": "55.5",
            "lang": "eng+deu",
            "profile": "sparse",
            "noise_suppression": False,
        }
    }
    got = ocr_config(cfg)
    assert got["enabled"] is True
    assert got["timeout_sec"] == 45
    assert got["min_chars_trigger"] == 110
    assert got["max_pages"] == 15
    assert got["min_output_chars"] == 250
    assert got["min_gain_chars"] == 60
    assert got["min_confidence"] == 55.5
    assert got["lang"] == "eng+deu"
    assert got["profile"] == "sparse"
    assert got["noise_suppression"] is False


def test_ocr_config_drops_invalid_profile_and_clamps() -> None:
    cfg = {
        "ocr": {
            "timeout_sec": 0,
            "min_chars_trigger": -2,
            "max_pages": -1,
            "min_output_chars": -9,
            "min_gain_chars": -40,
            "min_confidence": 200.0,
            "profile": "invalid",
        }
    }
    got = ocr_config(cfg)
    assert got["timeout_sec"] == 1
    assert got["min_chars_trigger"] == 1
    assert got["max_pages"] == 1
    assert got["min_output_chars"] == 1
    assert got["min_gain_chars"] == 0
    assert got["min_confidence"] == 100.0
    assert "profile" not in got
