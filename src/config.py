"""Tiny-GPT model configuration."""

from __future__ import annotations

__all__ = ["TinyGPTConfig"]

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import yaml


@dataclass
class TinyGPTConfig:
    batch_size: int = 16
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    max_steps: int = 1000
    eval_every: int = 100
    eval_iters: int = 20
    gen_tokens: int = 256
    temperature: float = 0.9
    top_k: int = 40
    seed: int = 1337

    @classmethod
    def from_path(cls, path: str | Path) -> "TinyGPTConfig":
        p = Path(path)
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
