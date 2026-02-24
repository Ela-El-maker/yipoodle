import torch
import pytest

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import TinyGPTConfig
from src.model.gpt import MiniGPT
from src.tokenizer import CharTokenizer


def test_save_load_deterministic_generation(tmp_path) -> None:
    cfg = TinyGPTConfig(block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    tok = CharTokenizer.from_text("hello world")
    model = MiniGPT(tok.vocab_size, cfg)

    ckpt = tmp_path / "m.pt"
    save_checkpoint(ckpt, model, cfg, tok)

    device = torch.device("cpu")
    model2, _, tok2 = load_checkpoint(ckpt, device)
    start = torch.tensor([[tok2.encode("he")[0], tok2.encode("he")[1]]], dtype=torch.long)
    out1 = model.generate(start.clone(), max_new_tokens=6, deterministic=True)
    out2 = model2.generate(start.clone(), max_new_tokens=6, deterministic=True)
    assert torch.equal(out1, out2)


def test_load_checkpoint_rejects_invalid_payload(tmp_path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"cfg": {}, "chars": "abc"}, bad)
    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(bad, torch.device("cpu"))
