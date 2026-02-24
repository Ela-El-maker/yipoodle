import json
from pathlib import Path

import numpy as np

from src.apps import vector_index
from src.core.schemas import SnippetRecord


class _FakeSTModel:
    def encode(self, texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            # deterministic tiny embedding
            base = np.array([len(t), sum(ord(c) for c in t) % 97, t.count(" ") + 1], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(base) or 1.0
                base = base / norm
            vecs.append(base)
        return np.vstack(vecs)


class _FakeIndexFlatIP:
    def __init__(self, dim: int):
        self.dim = dim
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def add(self, emb: np.ndarray):
        self._mat = np.vstack([self._mat, emb.astype(np.float32)])

    @property
    def ntotal(self):
        return self._mat.shape[0]

    def search(self, q: np.ndarray, k: int):
        sims = self._mat @ q[0]
        order = np.argsort(-sims)[:k]
        return sims[order].reshape(1, -1), order.reshape(1, -1)


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP

    @staticmethod
    def write_index(index: _FakeIndexFlatIP, path: str):
        payload = {"dim": index.dim, "mat": index._mat.tolist()}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def read_index(path: str):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        out = _FakeIndexFlatIP(int(payload["dim"]))
        out._mat = np.array(payload["mat"], dtype=np.float32)
        return out


def test_build_save_load_and_query_vector_index(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(vector_index, "_import_vector_deps", lambda: (_FakeFaiss, object))
    monkeypatch.setattr(vector_index, "_get_embedding_model", lambda _name: _FakeSTModel())

    snippets = [
        SnippetRecord(snippet_id="P1:S1", paper_id="P1", section="results", text="mobile segmentation", token_count=2),
        SnippetRecord(snippet_id="P2:S1", paper_id="P2", section="results", text="background matting", token_count=2),
    ]
    bundle = vector_index.build_vector_index(snippets, model_name="fake/model", batch_size=2)
    assert bundle.dimension == 3
    assert len(bundle.snippet_ids) == 2

    idx_path = tmp_path / "idx.faiss"
    meta_path = tmp_path / "idx.vector_meta.json"
    vector_index.save_vector_index(bundle, str(idx_path), str(meta_path))
    assert idx_path.exists()
    assert meta_path.exists()

    loaded = vector_index.load_vector_index(str(idx_path), str(meta_path))
    got = vector_index.query_vector_index(loaded, question="mobile", top_k=1, model_name_override="fake/model")
    assert len(got) == 1
    assert got[0][0] in {"P1:S1", "P2:S1"}
