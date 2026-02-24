from src.apps.retrieval import SimpleBM25Index
from src.core.schemas import SnippetRecord


def test_quality_prior_downweights_poor_extraction() -> None:
    snippets = [
        SnippetRecord(
            snippet_id="Pp1:S1",
            paper_id="p1",
            section="results",
            text="mobile segmentation boundary quality method",
            token_count=5,
            extraction_quality_score=0.2,
        ),
        SnippetRecord(
            snippet_id="Pp2:S1",
            paper_id="p2",
            section="results",
            text="mobile segmentation boundary quality method",
            token_count=5,
            extraction_quality_score=1.0,
        ),
    ]
    idx = SimpleBM25Index.build(snippets)
    ranked = idx.query_scored(
        question="mobile segmentation boundary quality",
        top_k=2,
        quality_prior_weight=0.15,
        max_per_paper=None,
    )
    assert ranked[0][0].snippet_id == "Pp2:S1"
    assert ranked[1][0].snippet_id == "Pp1:S1"
