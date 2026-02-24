import pytest

from src.apps.vector_backend import (
    FaissFlatBackend,
    FaissHNSWBackend,
    FaissIVFFlatBackend,
    get_faiss_backend,
)


class _FakeFaiss:
    pass


def test_backend_factory_routes_supported_types() -> None:
    faiss = _FakeFaiss()
    assert isinstance(get_faiss_backend(faiss, "flat"), FaissFlatBackend)
    assert isinstance(get_faiss_backend(faiss, "ivf_flat"), FaissIVFFlatBackend)
    assert isinstance(get_faiss_backend(faiss, "hnsw"), FaissHNSWBackend)


def test_backend_factory_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported vector index type"):
        get_faiss_backend(_FakeFaiss(), "nope")
