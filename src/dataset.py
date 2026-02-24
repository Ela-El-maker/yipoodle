from __future__ import annotations

from dataclasses import dataclass

import torch

from src.tokenizer import CharTokenizer


@dataclass
class TextDataset:
    tokenizer: CharTokenizer
    train_data: torch.Tensor
    val_data: torch.Tensor
    block_size: int

    @classmethod
    def from_text(cls, text: str, block_size: int, split_ratio: float = 0.9) -> "TextDataset":
        tok = CharTokenizer.from_text(text)
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        n = int(split_ratio * len(ids))
        return cls(tokenizer=tok, train_data=ids[:n], val_data=ids[n:], block_size=block_size)

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Dataset too small for configured block_size")
        idx = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in idx])
        return x.to(device), y.to(device)
