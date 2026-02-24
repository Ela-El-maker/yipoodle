import torch

from src.dataset import TextDataset


def test_batch_shapes_and_shift() -> None:
    ds = TextDataset.from_text("abcdefghijklmnopqrstuvwxyz" * 20, block_size=8)
    x, y = ds.get_batch("train", batch_size=4, device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
