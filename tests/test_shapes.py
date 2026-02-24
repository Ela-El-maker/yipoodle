import torch

from src.config import TinyGPTConfig
from src.model.gpt import MiniGPT


def test_forward_shape_and_loss_finite() -> None:
    cfg = TinyGPTConfig(block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    model = MiniGPT(vocab_size=30, cfg=cfg)
    idx = torch.randint(0, 30, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 30)
    assert loss is not None
    assert torch.isfinite(loss)


def test_causal_mask_behavior() -> None:
    cfg = TinyGPTConfig(block_size=8, n_embd=32, n_head=4, n_layer=1, dropout=0.0)
    model = MiniGPT(vocab_size=20, cfg=cfg)
    model.eval()

    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    b = torch.tensor([[1, 2, 3, 4, 9, 9, 9, 9]])

    logits_a, _ = model(a)
    logits_b, _ = model(b)

    assert torch.allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5)
