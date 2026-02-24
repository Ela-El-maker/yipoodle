from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from src.config import TinyGPTConfig
from src.model.gpt import MiniGPT
from src.tokenizer import CharTokenizer


def save_checkpoint(path: str | Path, model: MiniGPT, cfg: TinyGPTConfig, tokenizer: CharTokenizer) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
            "chars": tokenizer.chars,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[MiniGPT, TinyGPTConfig, CharTokenizer]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Secure checkpoint loading requires torch.load(..., weights_only=True). "
            "Your torch version appears too old. Upgrade torch to >=2.1."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint payload: expected dict")
    required = {"model_state", "cfg", "chars"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Invalid checkpoint payload: missing keys {sorted(missing)}")
    if not isinstance(payload.get("cfg"), dict):
        raise ValueError("Invalid checkpoint payload: cfg must be a dict")
    chars = payload.get("chars")
    if not isinstance(chars, list) or not all(isinstance(c, str) and len(c) == 1 for c in chars):
        raise ValueError("Invalid checkpoint payload: chars must be a list[str]")
    if not isinstance(payload.get("model_state"), dict):
        raise ValueError("Invalid checkpoint payload: model_state must be a dict")
    cfg = TinyGPTConfig(**payload["cfg"])
    tok = CharTokenizer(chars=chars)
    model = MiniGPT(tok.vocab_size, cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, cfg, tok
