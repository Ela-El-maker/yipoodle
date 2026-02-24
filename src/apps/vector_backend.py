from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VectorBuildConfig:
    model_name: str
    batch_size: int = 64
    index_type: str = "flat"  # flat|ivf_flat|hnsw
    nlist: int = 1024
    m: int = 32
    ef_construction: int = 200
    train_sample_size: int = 200000
    shards: int = 1


@dataclass
class VectorQueryConfig:
    top_k: int
    nprobe: int = 16
    ef_search: int = 64


@dataclass
class VectorBuildArtifact:
    index: Any
    snippet_ids: list[str]
    embedding_model: str
    dimension: int
    index_type: str
    ann_params: dict[str, int | float]
    shard_count: int
    build_stats: dict[str, int | float]


class VectorBackend:
    def build(self, embeddings, snippet_ids: list[str], config: VectorBuildConfig) -> VectorBuildArtifact:
        raise NotImplementedError

    def query(self, loaded_index: Any, question_vec, config: VectorQueryConfig) -> list[tuple[int, float]]:
        raise NotImplementedError


class FaissFlatBackend(VectorBackend):
    def __init__(self, faiss: Any):
        self.faiss = faiss

    def build(self, embeddings, snippet_ids: list[str], config: VectorBuildConfig) -> VectorBuildArtifact:
        dim = int(embeddings.shape[1])
        index = self.faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        return VectorBuildArtifact(
            index=index,
            snippet_ids=snippet_ids,
            embedding_model=config.model_name,
            dimension=dim,
            index_type="flat",
            ann_params={},
            shard_count=1,
            build_stats={"train_size": 0},
        )

    def query(self, loaded_index: Any, question_vec, config: VectorQueryConfig) -> list[tuple[int, float]]:
        k = max(1, int(config.top_k))
        scores, ids = loaded_index.search(question_vec, k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))


class FaissIVFFlatBackend(VectorBackend):
    def __init__(self, faiss: Any):
        self.faiss = faiss

    def build(self, embeddings, snippet_ids: list[str], config: VectorBuildConfig) -> VectorBuildArtifact:
        dim = int(embeddings.shape[1])
        nvec = int(embeddings.shape[0])
        nlist = max(1, min(int(config.nlist), max(1, nvec // 4)))
        quantizer = self.faiss.IndexFlatIP(dim)
        index = self.faiss.IndexIVFFlat(quantizer, dim, nlist, self.faiss.METRIC_INNER_PRODUCT)
        train_n = min(nvec, max(1, int(config.train_sample_size)))
        index.train(embeddings[:train_n].astype("float32"))
        index.add(embeddings.astype("float32"))
        return VectorBuildArtifact(
            index=index,
            snippet_ids=snippet_ids,
            embedding_model=config.model_name,
            dimension=dim,
            index_type="ivf_flat",
            ann_params={"nlist": int(nlist)},
            shard_count=1,
            build_stats={"train_size": int(train_n)},
        )

    def query(self, loaded_index: Any, question_vec, config: VectorQueryConfig) -> list[tuple[int, float]]:
        loaded_index.nprobe = max(1, int(config.nprobe))
        k = max(1, int(config.top_k))
        scores, ids = loaded_index.search(question_vec, k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))


class FaissHNSWBackend(VectorBackend):
    def __init__(self, faiss: Any):
        self.faiss = faiss

    def build(self, embeddings, snippet_ids: list[str], config: VectorBuildConfig) -> VectorBuildArtifact:
        dim = int(embeddings.shape[1])
        m = max(4, int(config.m))
        index = self.faiss.IndexHNSWFlat(dim, m, self.faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = max(8, int(config.ef_construction))
        index.add(embeddings.astype("float32"))
        return VectorBuildArtifact(
            index=index,
            snippet_ids=snippet_ids,
            embedding_model=config.model_name,
            dimension=dim,
            index_type="hnsw",
            ann_params={"m": int(m), "ef_construction": int(index.hnsw.efConstruction)},
            shard_count=1,
            build_stats={"train_size": 0},
        )

    def query(self, loaded_index: Any, question_vec, config: VectorQueryConfig) -> list[tuple[int, float]]:
        loaded_index.hnsw.efSearch = max(8, int(config.ef_search))
        k = max(1, int(config.top_k))
        scores, ids = loaded_index.search(question_vec, k)
        return list(zip(ids[0].tolist(), scores[0].tolist()))


def get_faiss_backend(faiss: Any, index_type: str) -> VectorBackend:
    t = str(index_type or "flat").lower()
    if t == "flat":
        return FaissFlatBackend(faiss)
    if t == "ivf_flat":
        return FaissIVFFlatBackend(faiss)
    if t == "hnsw":
        return FaissHNSWBackend(faiss)
    raise ValueError(f"Unsupported vector index type: {index_type}")
