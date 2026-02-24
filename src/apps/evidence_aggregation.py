"""Cross-document evidence aggregation.

Clusters :class:`EvidenceItem` instances by semantic similarity, then
labels each cluster with a consensus signal:

* **consensus** – items agree (high pairwise support, low contradiction).
* **mixed**     – partial agreement with some disagreement.
* **conflict**  – items contradict each other.

The module is designed to slot between evidence retrieval and synthesis
inside :func:`research_copilot.build_research_report`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np

from src.core.schemas import EvidenceItem, EvidencePack

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_CONTRADICTION_THRESHOLD = 0.40

ConsensusLabel = Literal["consensus", "mixed", "conflict"]

# The same negation set used in kb_confidence & semantic_validation.
_NEGATION_TOKENS = frozenset({
    "no", "not", "never", "without", "none", "cannot",
    "can't", "dont", "don't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "doesn't", "didn't",
})

_DIRECTION_PAIRS = [
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("higher", "lower"),
    ("more", "less"),
    ("improve", "worsen"),
    ("better", "worse"),
    ("gain", "loss"),
    ("positive", "negative"),
    ("faster", "slower"),
    ("larger", "smaller"),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCluster:
    """A group of semantically-similar evidence items."""

    cluster_id: int
    items: list[EvidenceItem] = field(default_factory=list)
    label: ConsensusLabel = "consensus"
    consensus_score: float = 1.0
    representative_text: str = ""
    paper_ids: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def cross_document(self) -> bool:
        """True when the cluster spans more than one paper."""
        return len(set(self.paper_ids)) > 1


@dataclass
class AggregationResult:
    """Output of the evidence aggregation pipeline."""

    clusters: list[EvidenceCluster] = field(default_factory=list)
    total_items: int = 0
    cross_document_clusters: int = 0
    consensus_clusters: int = 0
    conflict_clusters: int = 0
    mixed_clusters: int = 0

    @property
    def is_empty(self) -> bool:
        return len(self.clusters) == 0


# ---------------------------------------------------------------------------
# Embedding helpers (reuse the project's sentence-transformers infra)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Load a sentence-transformers model (cached)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required for evidence aggregation"
        ) from exc
    return SentenceTransformer(model_name, local_files_only=True)


def _encode_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Encode *texts* into L2-normalised vectors (cosine sim = dot product)."""
    model = _load_model(model_name)
    vecs: np.ndarray = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs


# ---------------------------------------------------------------------------
# Lightweight heuristic contradiction detector
# ---------------------------------------------------------------------------

def _heuristic_contradiction(a: str, b: str) -> float:
    """Return a 0–1 contradiction estimate between two text snippets.

    Combines negation-word mismatch and directional-language conflict.
    Mirrors the approach in ``kb_confidence.contradiction_score`` and
    ``semantic_validation._contradiction_proxy`` but is self-contained.
    """
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))

    # Word overlap guard — if texts share < 25 % vocabulary they are
    # probably about different topics rather than contradicting each other.
    overlap = len(wa & wb) / max(1, len(wa | wb))
    if overlap < 0.25:
        return 0.0

    score = 0.0

    # Negation mismatch
    a_neg = bool(wa & _NEGATION_TOKENS)
    b_neg = bool(wb & _NEGATION_TOKENS)
    if a_neg != b_neg:
        score += 0.50

    # Direction conflict
    for up, down in _DIRECTION_PAIRS:
        a_up = up in wa
        a_down = down in wa
        b_up = up in wb
        b_down = down in wb
        if (a_up and b_down) or (a_down and b_up):
            score += 0.35
            break  # one pair is enough

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Agglomerative clustering (no scikit-learn dependency)
# ---------------------------------------------------------------------------

def _agglomerative_cluster(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Single-linkage agglomerative clustering by cosine similarity.

    Returns a list of clusters, where each cluster is a list of indices.
    Two items are merged if their similarity ≥ *threshold*.
    """
    n = similarity_matrix.shape[0]
    # Union-Find
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Consensus labelling
# ---------------------------------------------------------------------------

def _label_cluster(
    items: list[EvidenceItem],
    sim_matrix: np.ndarray,
    indices: list[int],
    contradiction_threshold: float,
) -> tuple[ConsensusLabel, float]:
    """Compute consensus label for one cluster.

    Returns ``(label, consensus_score)`` where *consensus_score* ∈ [0, 1].
    """
    if len(indices) <= 1:
        return "consensus", 1.0

    # Collect pairwise contradiction scores
    contradiction_scores: list[float] = []
    support_scores: list[float] = []
    for ii, idx_a in enumerate(indices):
        for idx_b in indices[ii + 1:]:
            c = _heuristic_contradiction(items[idx_a].text, items[idx_b].text)
            contradiction_scores.append(c)
            support_scores.append(float(sim_matrix[idx_a, idx_b]))

    avg_contradiction = sum(contradiction_scores) / len(contradiction_scores)
    avg_support = sum(support_scores) / len(support_scores)

    if avg_contradiction >= contradiction_threshold:
        label: ConsensusLabel = "conflict"
    elif avg_contradiction >= contradiction_threshold * 0.5:
        label = "mixed"
    else:
        label = "consensus"

    # consensus_score: high support + low contradiction → near 1.0
    consensus_score = max(0.0, min(1.0, avg_support - avg_contradiction))
    return label, round(consensus_score, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_evidence(
    evidence: EvidencePack,
    *,
    model_name: str = DEFAULT_CLUSTER_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    max_items: int = 20,
) -> AggregationResult:
    """Cluster evidence items by semantic similarity and label consensus.

    Parameters
    ----------
    evidence:
        The evidence pack from retrieval.
    model_name:
        Sentence-transformer model for embedding.
    similarity_threshold:
        Minimum cosine similarity to group items into the same cluster.
    contradiction_threshold:
        Average heuristic-contradiction score above which a cluster is
        labelled *conflict*.
    max_items:
        Cap on evidence items to process (keeps latency bounded).

    Returns
    -------
    AggregationResult
        Clusters with consensus/conflict labels and summary statistics.
    """
    items = evidence.items[:max_items]
    if not items:
        return AggregationResult()

    if len(items) == 1:
        cluster = EvidenceCluster(
            cluster_id=0,
            items=list(items),
            label="consensus",
            consensus_score=1.0,
            representative_text=items[0].text,
            paper_ids=[items[0].paper_id],
        )
        return AggregationResult(
            clusters=[cluster],
            total_items=1,
            consensus_clusters=1,
        )

    # 1. Encode all evidence texts
    texts = [it.text for it in items]
    vecs = _encode_texts(texts, model_name)

    # 2. Pairwise cosine similarity (vecs are L2-normalised)
    sim_matrix = vecs @ vecs.T

    # 3. Cluster
    raw_clusters = _agglomerative_cluster(sim_matrix, similarity_threshold)
    log.info(
        "Evidence aggregation: %d items → %d clusters (threshold=%.2f)",
        len(items), len(raw_clusters), similarity_threshold,
    )

    # 4. Label each cluster
    clusters: list[EvidenceCluster] = []
    for cid, indices in enumerate(raw_clusters):
        cluster_items = [items[i] for i in indices]
        paper_ids = [it.paper_id for it in cluster_items]

        label, consensus_score = _label_cluster(
            items, sim_matrix, indices, contradiction_threshold,
        )

        # Representative = highest-scored item
        best_idx = max(indices, key=lambda i: items[i].score)
        representative = items[best_idx].text

        clusters.append(EvidenceCluster(
            cluster_id=cid,
            items=cluster_items,
            label=label,
            consensus_score=consensus_score,
            representative_text=representative,
            paper_ids=paper_ids,
        ))

    # Sort: consensus first, then mixed, then conflict; within groups by size desc
    order = {"consensus": 0, "mixed": 1, "conflict": 2}
    clusters.sort(key=lambda c: (order.get(c.label, 9), -c.size))

    cross_doc = sum(1 for c in clusters if c.cross_document)
    consensus_n = sum(1 for c in clusters if c.label == "consensus")
    conflict_n = sum(1 for c in clusters if c.label == "conflict")
    mixed_n = sum(1 for c in clusters if c.label == "mixed")

    return AggregationResult(
        clusters=clusters,
        total_items=len(items),
        cross_document_clusters=cross_doc,
        consensus_clusters=consensus_n,
        conflict_clusters=conflict_n,
        mixed_clusters=mixed_n,
    )


# ---------------------------------------------------------------------------
# Synthesis helper — enhanced synthesis that leverages clusters
# ---------------------------------------------------------------------------

def synthesize_from_clusters(
    result: AggregationResult,
    question: str,
) -> dict:
    """Produce synthesis text, gaps, experiments, and citations from clusters.

    Returns a dict with keys:
    ``synthesis``, ``gaps``, ``experiments``, ``citations``,
    ``aggregation_summary``.
    """
    from src.core.schemas import ExperimentProposal

    if result.is_empty:
        return {
            "synthesis": "Not found in sources.",
            "gaps": ["Not found in sources."],
            "experiments": [],
            "citations": [],
            "aggregation_summary": {},
        }

    lines: list[str] = []
    gaps: list[str] = []
    experiments: list[ExperimentProposal] = []
    citations: list[str] = []

    for cluster in result.clusters:
        if not cluster.items:
            continue

        paper_ids = sorted(set(cluster.paper_ids))
        n_papers = len(paper_ids)
        label_tag = cluster.label.upper()

        # Build cluster heading
        cluster_cits: list[str] = []
        for item in cluster.items:
            cit = f"({item.snippet_id})"
            cluster_cits.append(cit)
            citations.append(cit)

        cit_str = ", ".join(cluster_cits[:4])
        if len(cluster_cits) > 4:
            cit_str += f" +{len(cluster_cits) - 4} more"

        # Representative claim
        rep_short = " ".join(cluster.representative_text.split()[:24])

        if n_papers > 1:
            lines.append(
                f"[{label_tag} — {n_papers} papers] "
                f"{rep_short} ... {cit_str}"
            )
        else:
            lines.append(
                f"[{label_tag}] "
                f"{rep_short} ... {cit_str}"
            )

        # Conflict → gap
        if cluster.label == "conflict":
            gap_pids = ", ".join(paper_ids[:3])
            gaps.append(
                f"Conflicting evidence across {n_papers} paper(s) ({gap_pids}): "
                f"see {cit_str}"
            )
            experiments.append(ExperimentProposal(
                proposal=f"Resolve conflict: replicate findings from {gap_pids} {cit_str}",
                citations=cluster_cits[:4],
            ))

        # Mixed → note
        if cluster.label == "mixed":
            gaps.append(
                f"Mixed evidence (consensus_score={cluster.consensus_score:.2f}) — "
                f"further investigation needed {cit_str}"
            )

        # Scan for limitation / future work
        for item in cluster.items:
            lower = item.text.lower()
            if "limitation" in lower or "future work" in lower or "fails" in lower:
                cit = f"({item.snippet_id})"
                gaps.append(f"Potential gap around {item.section} {cit}")
                experiments.append(ExperimentProposal(
                    proposal=f"Test an ablation around {item.section} features {cit}",
                    citations=[cit],
                ))

    # Fallback gap
    if not gaps and citations:
        first_cit = citations[0]
        gaps.append(f"No explicit limitations extracted; gather more diverse evidence {first_cit}")
        experiments.append(ExperimentProposal(
            proposal=f"Run baseline + one robustness ablation {first_cit}",
            citations=[first_cit],
        ))

    summary = {
        "total_clusters": len(result.clusters),
        "cross_document_clusters": result.cross_document_clusters,
        "consensus_clusters": result.consensus_clusters,
        "conflict_clusters": result.conflict_clusters,
        "mixed_clusters": result.mixed_clusters,
        "total_items_aggregated": result.total_items,
    }

    return {
        "synthesis": "\n".join(lines),
        "gaps": gaps,
        "experiments": experiments,
        "citations": sorted(set(citations)),
        "aggregation_summary": summary,
    }


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_aggregation_markdown(result: AggregationResult) -> str:
    """Render a Markdown summary of the aggregation result."""
    lines: list[str] = [
        "## Evidence Aggregation",
        "",
        f"- **Total evidence items**: {result.total_items}",
        f"- **Clusters**: {len(result.clusters)}",
        f"- **Cross-document clusters**: {result.cross_document_clusters}",
        f"- **Consensus**: {result.consensus_clusters}",
        f"- **Mixed**: {result.mixed_clusters}",
        f"- **Conflict**: {result.conflict_clusters}",
        "",
    ]
    for cluster in result.clusters:
        papers = sorted(set(cluster.paper_ids))
        rep_short = " ".join(cluster.representative_text.split()[:30])
        label = cluster.label.upper()
        lines.append(
            f"### Cluster {cluster.cluster_id} [{label}] "
            f"({cluster.size} items, {len(papers)} papers, "
            f"consensus={cluster.consensus_score:.2f})"
        )
        lines.append(f"> {rep_short} …")
        for item in cluster.items:
            lines.append(f"  - `{item.snippet_id}` (score={item.score:.3f}, {item.section})")
        lines.append("")
    return "\n".join(lines)
