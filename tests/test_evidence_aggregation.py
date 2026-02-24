"""Tests for cross-document evidence aggregation (Feature #3).

Covers:
- Heuristic contradiction detection
- Agglomerative clustering
- Cluster consensus labelling
- Full aggregation pipeline
- Enhanced synthesis from clusters
- Markdown rendering
- Integration with build_research_report
- Edge cases: empty evidence, single item, identical items
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.apps.evidence_aggregation import (
    AggregationResult,
    DEFAULT_CLUSTER_MODEL,
    EvidenceCluster,
    _agglomerative_cluster,
    _heuristic_contradiction,
    _label_cluster,
    aggregate_evidence,
    render_aggregation_markdown,
    synthesize_from_clusters,
)
from src.core.schemas import (
    EvidenceItem,
    EvidencePack,
    ExperimentProposal,
    ResearchReport,
    ShortlistItem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_item(
    paper_id: str = "doi:10/test1",
    snippet_id: str = "Ptest1:S1",
    text: str = "Attention mechanisms scale quadratically with sequence length.",
    section: str = "method",
    score: float = 0.9,
    paper_year: int | None = 2023,
) -> EvidenceItem:
    return EvidenceItem(
        paper_id=paper_id,
        snippet_id=snippet_id,
        score=score,
        section=section,
        text=text,
        paper_year=paper_year,
        paper_venue="NeurIPS",
    )


@pytest.fixture()
def diverse_evidence() -> EvidencePack:
    """Evidence pack with items spanning different topics and papers."""
    return EvidencePack(
        question="How does attention scale?",
        items=[
            # Cluster: attention scaling (2 papers agree)
            _make_item(
                paper_id="doi:10/p1", snippet_id="Pp1:S1",
                text="Attention mechanisms have quadratic computational complexity O(n^2) in sequence length.",
                score=0.95,
            ),
            _make_item(
                paper_id="doi:10/p2", snippet_id="Pp2:S1",
                text="Self-attention complexity grows quadratically with respect to the input sequence length.",
                score=0.90,
            ),
            # Cluster: linear attention (different topic)
            _make_item(
                paper_id="doi:10/p3", snippet_id="Pp3:S1",
                text="Linear attention reduces the complexity from quadratic to linear time.",
                section="results", score=0.85,
            ),
            # Cluster: contradiction (one says improvement, other says no improvement)
            _make_item(
                paper_id="doi:10/p4", snippet_id="Pp4:S1",
                text="The proposed method significantly increased accuracy on all benchmarks.",
                section="results", score=0.80,
            ),
            _make_item(
                paper_id="doi:10/p5", snippet_id="Pp5:S1",
                text="The proposed method did not increase accuracy and showed decreased performance.",
                section="results", score=0.75,
            ),
        ],
    )


@pytest.fixture()
def single_item_evidence() -> EvidencePack:
    return EvidencePack(
        question="What is attention?",
        items=[_make_item()],
    )


@pytest.fixture()
def empty_evidence() -> EvidencePack:
    return EvidencePack(question="Nothing?", items=[])


# ---------------------------------------------------------------------------
# Mock encoder — avoid loading the real model in unit tests
# ---------------------------------------------------------------------------


def _mock_encode(texts, **kwargs):
    """Produce deterministic normalised vectors from text content.

    We use a simple hash-based scheme so that semantically similar texts
    (sharing many words) have higher dot-product similarity.
    """
    vecs = []
    for text in texts:
        words = set(text.lower().split())
        # 64-dim vector where each dim is 1.0 if a word hash maps there
        v = np.zeros(64, dtype=np.float32)
        for w in words:
            idx = hash(w) % 64
            v[idx] += 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


class _MockModel:
    def encode(self, texts, **kwargs):
        return _mock_encode(texts, **kwargs)


@pytest.fixture(autouse=True)
def _patch_model():
    """Patch the sentence-transformer model loading for all tests."""
    with patch(
        "src.apps.evidence_aggregation._load_model",
        return_value=_MockModel(),
    ):
        yield


# ---------------------------------------------------------------------------
# TestHeuristicContradiction
# ---------------------------------------------------------------------------


class TestHeuristicContradiction:
    def test_no_contradiction(self) -> None:
        a = "Attention mechanisms scale quadratically."
        b = "Self-attention has quadratic complexity."
        score = _heuristic_contradiction(a, b)
        assert score < 0.3

    def test_negation_contradiction(self) -> None:
        a = "The method increased accuracy."
        b = "The method did not increase accuracy."
        score = _heuristic_contradiction(a, b)
        assert score >= 0.4

    def test_direction_conflict(self) -> None:
        a = "Performance increased significantly."
        b = "Performance decreased significantly."
        score = _heuristic_contradiction(a, b)
        assert score >= 0.3

    def test_low_overlap_no_contradiction(self) -> None:
        a = "Quantum computing uses qubits."
        b = "Baking bread requires flour."
        score = _heuristic_contradiction(a, b)
        assert score == 0.0

    def test_empty_strings(self) -> None:
        assert _heuristic_contradiction("", "") == 0.0

    def test_identical_text(self) -> None:
        text = "The model achieves state of the art results."
        score = _heuristic_contradiction(text, text)
        assert score < 0.1  # Should not self-contradict


# ---------------------------------------------------------------------------
# TestAgglomerativeClustering
# ---------------------------------------------------------------------------


class TestAgglomerativeClustering:
    def test_all_similar(self) -> None:
        sim = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        clusters = _agglomerative_cluster(sim, threshold=0.7)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_all_dissimilar(self) -> None:
        sim = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.15],
            [0.2, 0.15, 1.0],
        ])
        clusters = _agglomerative_cluster(sim, threshold=0.5)
        assert len(clusters) == 3

    def test_two_clusters(self) -> None:
        sim = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0],
        ])
        clusters = _agglomerative_cluster(sim, threshold=0.5)
        assert len(clusters) == 2
        cluster_sets = [frozenset(c) for c in clusters]
        assert frozenset({0, 1}) in cluster_sets
        assert frozenset({2, 3}) in cluster_sets

    def test_single_item(self) -> None:
        sim = np.array([[1.0]])
        clusters = _agglomerative_cluster(sim, threshold=0.5)
        assert len(clusters) == 1
        assert clusters[0] == [0]


# ---------------------------------------------------------------------------
# TestClusterLabelling
# ---------------------------------------------------------------------------


class TestClusterLabelling:
    def _item(self, text: str) -> EvidenceItem:
        return _make_item(text=text)

    def test_consensus_label(self) -> None:
        items = [
            self._item("Attention is efficient for short sequences."),
            self._item("Attention works well for short sequences."),
        ]
        sim = np.array([[1.0, 0.9], [0.9, 1.0]])
        label, score = _label_cluster(items, sim, [0, 1], contradiction_threshold=0.40)
        assert label == "consensus"
        assert score > 0.0

    def test_conflict_label(self) -> None:
        # Texts share high word overlap so the overlap guard passes,
        # but one has negation → contradiction heuristic fires.
        items = [
            self._item("The model accuracy increased on the benchmark dataset results."),
            self._item("The model accuracy did not increase on the benchmark dataset results."),
        ]
        sim = np.array([[1.0, 0.6], [0.6, 1.0]])
        label, score = _label_cluster(items, sim, [0, 1], contradiction_threshold=0.40)
        assert label in ("conflict", "mixed")

    def test_single_item_cluster(self) -> None:
        items = [self._item("Some text")]
        sim = np.array([[1.0]])
        label, score = _label_cluster(items, sim, [0], contradiction_threshold=0.40)
        assert label == "consensus"
        assert score == 1.0


# ---------------------------------------------------------------------------
# TestAggregateEvidence
# ---------------------------------------------------------------------------


class TestAggregateEvidence:
    def test_empty_evidence(self, empty_evidence: EvidencePack) -> None:
        result = aggregate_evidence(empty_evidence)
        assert result.is_empty
        assert result.total_items == 0

    def test_single_item(self, single_item_evidence: EvidencePack) -> None:
        result = aggregate_evidence(single_item_evidence)
        assert len(result.clusters) == 1
        assert result.total_items == 1
        assert result.consensus_clusters == 1

    def test_diverse_evidence_clusters(self, diverse_evidence: EvidencePack) -> None:
        result = aggregate_evidence(diverse_evidence, similarity_threshold=0.3)
        assert result.total_items == 5
        assert len(result.clusters) >= 1
        # Should have at least some cross-document clusters
        total_items_in_clusters = sum(c.size for c in result.clusters)
        assert total_items_in_clusters == 5

    def test_cluster_ids_unique(self, diverse_evidence: EvidencePack) -> None:
        result = aggregate_evidence(diverse_evidence, similarity_threshold=0.3)
        ids = [c.cluster_id for c in result.clusters]
        assert len(ids) == len(set(ids))

    def test_max_items_cap(self, diverse_evidence: EvidencePack) -> None:
        result = aggregate_evidence(diverse_evidence, max_items=2)
        assert result.total_items == 2

    def test_aggregation_result_counts(self, diverse_evidence: EvidencePack) -> None:
        result = aggregate_evidence(diverse_evidence, similarity_threshold=0.3)
        assert result.consensus_clusters + result.conflict_clusters + result.mixed_clusters == len(result.clusters)

    def test_cluster_has_paper_ids(self, diverse_evidence: EvidencePack) -> None:
        result = aggregate_evidence(diverse_evidence, similarity_threshold=0.3)
        for cluster in result.clusters:
            assert len(cluster.paper_ids) == len(cluster.items)

    def test_representative_text_set(self, diverse_evidence: EvidencePack) -> None:
        result = aggregate_evidence(diverse_evidence, similarity_threshold=0.3)
        for cluster in result.clusters:
            assert cluster.representative_text


# ---------------------------------------------------------------------------
# TestSynthesizeFromClusters
# ---------------------------------------------------------------------------


class TestSynthesizeFromClusters:
    def test_empty_result(self) -> None:
        result = AggregationResult()
        out = synthesize_from_clusters(result, "test?")
        assert out["synthesis"] == "Not found in sources."
        assert len(out["gaps"]) > 0

    def test_consensus_cluster_synthesis(self) -> None:
        items = [
            _make_item(paper_id="doi:10/p1", snippet_id="Pp1:S1", text="Attention is quadratic."),
            _make_item(paper_id="doi:10/p2", snippet_id="Pp2:S1", text="Self-attention is O(n^2)."),
        ]
        cluster = EvidenceCluster(
            cluster_id=0,
            items=items,
            label="consensus",
            consensus_score=0.8,
            representative_text=items[0].text,
            paper_ids=[it.paper_id for it in items],
        )
        result = AggregationResult(
            clusters=[cluster],
            total_items=2,
            consensus_clusters=1,
            cross_document_clusters=1,
        )
        out = synthesize_from_clusters(result, "How does attention scale?")
        assert "CONSENSUS" in out["synthesis"]
        assert "2 papers" in out["synthesis"]
        assert len(out["citations"]) == 2

    def test_conflict_cluster_adds_gap(self) -> None:
        items = [
            _make_item(paper_id="doi:10/p4", snippet_id="Pp4:S1",
                       text="The method increased accuracy."),
            _make_item(paper_id="doi:10/p5", snippet_id="Pp5:S1",
                       text="The method did not increase accuracy."),
        ]
        cluster = EvidenceCluster(
            cluster_id=0,
            items=items,
            label="conflict",
            consensus_score=0.2,
            representative_text=items[0].text,
            paper_ids=[it.paper_id for it in items],
        )
        result = AggregationResult(
            clusters=[cluster],
            total_items=2,
            conflict_clusters=1,
        )
        out = synthesize_from_clusters(result, "Does the method work?")
        assert "CONFLICT" in out["synthesis"]
        assert any("Conflicting" in g for g in out["gaps"])
        assert len(out["experiments"]) >= 1

    def test_mixed_cluster_adds_gap(self) -> None:
        items = [
            _make_item(paper_id="doi:10/p1", snippet_id="Pp1:S1", text="Results show improvement."),
        ]
        cluster = EvidenceCluster(
            cluster_id=0, items=items, label="mixed",
            consensus_score=0.5, representative_text=items[0].text,
            paper_ids=["doi:10/p1"],
        )
        result = AggregationResult(clusters=[cluster], total_items=1, mixed_clusters=1)
        out = synthesize_from_clusters(result, "q")
        assert any("Mixed" in g for g in out["gaps"])

    def test_aggregation_summary_keys(self) -> None:
        cluster = EvidenceCluster(
            cluster_id=0, items=[_make_item()], label="consensus",
            consensus_score=1.0, representative_text="text",
            paper_ids=["doi:10/test1"],
        )
        result = AggregationResult(clusters=[cluster], total_items=1, consensus_clusters=1)
        out = synthesize_from_clusters(result, "q")
        summary = out["aggregation_summary"]
        assert "total_clusters" in summary
        assert "cross_document_clusters" in summary
        assert "consensus_clusters" in summary
        assert "conflict_clusters" in summary

    def test_limitation_text_creates_gap(self) -> None:
        item = _make_item(text="This approach has a limitation in scalability.")
        cluster = EvidenceCluster(
            cluster_id=0, items=[item], label="consensus",
            consensus_score=1.0, representative_text=item.text,
            paper_ids=[item.paper_id],
        )
        result = AggregationResult(clusters=[cluster], total_items=1, consensus_clusters=1)
        out = synthesize_from_clusters(result, "q")
        assert any("gap" in g.lower() or "limitation" in g.lower() for g in out["gaps"])


# ---------------------------------------------------------------------------
# TestRenderAggregationMarkdown
# ---------------------------------------------------------------------------


class TestRenderAggregationMarkdown:
    def test_empty_result(self) -> None:
        result = AggregationResult()
        md = render_aggregation_markdown(result)
        assert "Evidence Aggregation" in md
        assert "Clusters**: 0" in md

    def test_renders_cluster_details(self) -> None:
        items = [
            _make_item(paper_id="doi:10/p1", snippet_id="Pp1:S1", text="Text A"),
            _make_item(paper_id="doi:10/p2", snippet_id="Pp2:S1", text="Text B"),
        ]
        cluster = EvidenceCluster(
            cluster_id=0, items=items, label="consensus",
            consensus_score=0.85, representative_text="Text A",
            paper_ids=["doi:10/p1", "doi:10/p2"],
        )
        result = AggregationResult(
            clusters=[cluster], total_items=2,
            consensus_clusters=1, cross_document_clusters=1,
        )
        md = render_aggregation_markdown(result)
        assert "CONSENSUS" in md
        assert "Pp1:S1" in md
        assert "Pp2:S1" in md
        assert "consensus=0.85" in md


# ---------------------------------------------------------------------------
# TestEvidenceClusterProperties
# ---------------------------------------------------------------------------


class TestEvidenceClusterProperties:
    def test_cross_document(self) -> None:
        c = EvidenceCluster(
            cluster_id=0, items=[], label="consensus",
            paper_ids=["doi:10/p1", "doi:10/p2"],
        )
        assert c.cross_document is True

    def test_single_document(self) -> None:
        c = EvidenceCluster(
            cluster_id=0, items=[], label="consensus",
            paper_ids=["doi:10/p1", "doi:10/p1"],
        )
        assert c.cross_document is False

    def test_size(self) -> None:
        c = EvidenceCluster(
            cluster_id=0,
            items=[_make_item(), _make_item(snippet_id="Ptest1:S2")],
            label="consensus",
        )
        assert c.size == 2


# ---------------------------------------------------------------------------
# TestBuildResearchReportWithAggregation
# ---------------------------------------------------------------------------


class TestBuildResearchReportWithAggregation:
    def test_report_uses_aggregation_when_provided(self) -> None:
        from src.apps.research_copilot import build_research_report

        items = [
            _make_item(paper_id="doi:10/p1", snippet_id="Pp1:S1",
                       text="Attention has quadratic complexity."),
            _make_item(paper_id="doi:10/p2", snippet_id="Pp2:S1",
                       text="Self-attention scales as O(n^2)."),
        ]
        evidence = EvidencePack(question="How does attention scale?", items=items)
        cluster = EvidenceCluster(
            cluster_id=0, items=items, label="consensus",
            consensus_score=0.85, representative_text=items[0].text,
            paper_ids=[it.paper_id for it in items],
        )
        agg = AggregationResult(
            clusters=[cluster], total_items=2,
            consensus_clusters=1, cross_document_clusters=1,
        )
        report = build_research_report(
            "How does attention scale?", evidence, aggregation=agg,
        )
        assert "CONSENSUS" in report.synthesis
        assert len(report.citations) >= 1

    def test_report_falls_back_without_aggregation(self) -> None:
        from src.apps.research_copilot import build_research_report

        items = [
            _make_item(paper_id="doi:10/p1", snippet_id="Pp1:S1",
                       text="Some evidence text here."),
        ]
        evidence = EvidencePack(question="Test?", items=items)
        report = build_research_report("Test?", evidence, min_items=1, aggregation=None)
        assert "Claim:" in report.synthesis  # falls back to _synthesize

    def test_report_with_empty_aggregation(self) -> None:
        from src.apps.research_copilot import build_research_report

        items = [
            _make_item(paper_id="doi:10/p1", snippet_id="Pp1:S1",
                       text="Some evidence text here."),
        ]
        evidence = EvidencePack(question="Test?", items=items)
        empty_agg = AggregationResult()
        report = build_research_report("Test?", evidence, min_items=1, aggregation=empty_agg)
        # Empty aggregation → should fall back to _synthesize
        assert "Claim:" in report.synthesis


# ---------------------------------------------------------------------------
# TestRenderReportWithAggregation
# ---------------------------------------------------------------------------


class TestRenderReportWithAggregation:
    def test_markdown_includes_aggregation_section(self) -> None:
        from src.apps.research_copilot import render_report_markdown

        report = ResearchReport(
            question="Test?",
            synthesis="Some synthesis.",
            key_claims=["Claim 1"],
            gaps=["Gap 1"],
            experiments=[],
            citations=["(Pp1:S1)"],
        )
        cluster = EvidenceCluster(
            cluster_id=0, items=[_make_item()], label="consensus",
            consensus_score=1.0, representative_text="text",
            paper_ids=["doi:10/test1"],
        )
        agg = AggregationResult(clusters=[cluster], total_items=1, consensus_clusters=1)
        md = render_report_markdown(report, aggregation=agg)
        assert "Evidence Aggregation" in md

    def test_markdown_without_aggregation(self) -> None:
        from src.apps.research_copilot import render_report_markdown

        report = ResearchReport(
            question="Test?",
            synthesis="Some synthesis.",
            key_claims=[],
            gaps=[],
            experiments=[],
            citations=[],
        )
        md = render_report_markdown(report, aggregation=None)
        assert "Evidence Aggregation" not in md
