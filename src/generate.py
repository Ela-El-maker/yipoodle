from __future__ import annotations

import argparse

import torch

from src.checkpoint import load_checkpoint
from src.utils import get_device, set_seed


def generate_text(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int = 40,
    deterministic: bool = False,
    seed: int = 1337,
) -> str:
    set_seed(seed)
    device = get_device()
    model, _, tok = load_checkpoint(checkpoint, device)

    prompt_ids = tok.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        deterministic=deterministic,
    )
    return tok.decode(out[0].tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a tiny GPT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = generate_text(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
