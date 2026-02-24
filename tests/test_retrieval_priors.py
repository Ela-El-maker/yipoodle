from src.apps.retrieval import SimpleBM25Index
from src.core.schemas import SnippetRecord


def test_metadata_priors_affect_score_ordering() -> None:
    s_old = SnippetRecord(
        snippet_id="Pold:S1",
        paper_id="old",
        section="results",
        text="mobile segmentation model",
        token_count=3,
        paper_year=2016,
        paper_venue="Workshop",
        citation_count=2,
    )
    s_new = SnippetRecord(
        snippet_id="Pnew:S1",
        paper_id="new",
        section="results",
        text="mobile segmentation model",
        token_count=3,
        paper_year=2024,
        paper_venue="CVPR",
        citation_count=500,
    )
    idx = SimpleBM25Index.build([s_old, s_new])
    ev = idx.query("mobile segmentation", top_k=2, max_per_paper=1, use_metadata_priors=True)
    assert ev.items[0].paper_id == "new"
