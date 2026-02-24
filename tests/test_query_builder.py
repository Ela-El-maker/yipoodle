from src.apps.query_builder import build_query_plan


def test_cv_query_expansion_contains_related_terms() -> None:
    plan = build_query_plan("mobile segmentation", domain="computer_vision")
    assert "segmentation" in plan.term_boosts
    assert "matting" in plan.term_boosts
    assert "lightweight" in plan.term_boosts
    assert plan.section_weights["limitations"] > 1.0
