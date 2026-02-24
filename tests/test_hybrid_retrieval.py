from src.apps.retrieval import evidence_from_ranked, fuse_hybrid_scores, minmax_normalize_scores
from src.core.schemas import SnippetRecord


def test_minmax_normalize_constant_scores() -> None:
    scores = {"a": 3.0, "b": 3.0}
    out = minmax_normalize_scores(scores)
    assert out == {"a": 1.0, "b": 1.0}


def test_fuse_hybrid_scores_math() -> None:
    lexical = {"s1": 10.0, "s2": 1.0}
    vector = {"s2": 5.0, "s3": 4.0}
    fused = fuse_hybrid_scores(lexical, vector, alpha=0.6)
    assert set(fused.keys()) == {"s1", "s2", "s3"}
    assert fused["s1"] > fused["s3"]


def test_evidence_from_ranked_max_per_paper() -> None:
    s1 = SnippetRecord(snippet_id="P1:S1", paper_id="P1", section="results", text="a", token_count=1)
    s2 = SnippetRecord(snippet_id="P1:S2", paper_id="P1", section="results", text="b", token_count=1)
    s3 = SnippetRecord(snippet_id="P2:S1", paper_id="P2", section="results", text="c", token_count=1)
    ranked = [(s1, 1.0), (s2, 0.9), (s3, 0.8)]
    ev = evidence_from_ranked("q", ranked, top_k=3, max_per_paper=1)
    assert len(ev.items) == 2
    assert {i.paper_id for i in ev.items} == {"P1", "P2"}
