from src.apps.direct_answer import try_direct_answer


def test_arithmetic_direct_answer_simple() -> None:
    out = try_direct_answer("23 + 34 = ?", max_complexity=2)
    assert out.used is True
    assert out.answer_type == "arithmetic"
    assert out.value == "57"


def test_arithmetic_precedence_parentheses() -> None:
    out = try_direct_answer("(2 + 3) * 4", max_complexity=3)
    assert out.used is True
    assert out.value == "20"


def test_direct_answer_rejects_non_math_text() -> None:
    out = try_direct_answer("What is the weather in Nairobi today?", max_complexity=3)
    assert out.used is False


def test_direct_answer_complexity_cap() -> None:
    out = try_direct_answer("1+2+3+4+5", max_complexity=2)
    assert out.used is False
