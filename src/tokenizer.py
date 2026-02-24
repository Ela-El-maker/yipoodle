from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class CharTokenizer:
    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls(chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        lut = self.stoi
        return [lut[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        lut = self.itos
        return "".join(lut[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars=data["chars"])
