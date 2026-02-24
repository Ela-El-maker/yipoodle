from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import logging

import torch

log = logging.getLogger(__name__)

from src.config import TinyGPTConfig
from src.dataset import TextDataset
from src.model.gpt import MiniGPT
from src.checkpoint import save_checkpoint
from src.utils import get_device, set_seed, timestamp


def estimate_loss(model: MiniGPT, ds: TextDataset, cfg: TinyGPTConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    with torch.no_grad():
        for split in ("train", "val"):
            losses = []
            for _ in range(cfg.eval_iters):
                xb, yb = ds.get_batch(split, cfg.batch_size, device)
                _, loss = model(xb, yb)
                assert loss is not None
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train(data_path: str, out_dir: str, config_path: str | None = None) -> str:
    cfg = TinyGPTConfig.from_path(config_path) if config_path else TinyGPTConfig()
    set_seed(cfg.seed)
    device = get_device()

    text = Path(data_path).read_text(encoding="utf-8")
    ds = TextDataset.from_text(text, block_size=cfg.block_size)

    model = MiniGPT(ds.tokenizer.vocab_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    log_dir = Path(out_dir) / "logs"
    ckpt_dir = Path(out_dir) / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for step in range(1, cfg.max_steps + 1):
        xb, yb = ds.get_batch("train", cfg.batch_size, device)
        _, loss = model(xb, yb)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % cfg.eval_every == 0:
            losses = estimate_loss(model, ds, cfg, device)
            record = {"step": step, **losses}
            history.append(record)
            log.info("step=%d train=%.4f val=%.4f", step, losses['train'], losses['val'])

    run_id = timestamp()
    ckpt_path = ckpt_dir / f"mini_gpt_{run_id}.pt"
    save_checkpoint(ckpt_path, model, cfg, ds.tokenizer)
    (log_dir / f"train_{run_id}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (log_dir / f"config_{run_id}.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return str(ckpt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tiny GPT")
    parser.add_argument("--data", default="data/data.txt", help="Path to training text file")
    parser.add_argument("--out-dir", default="runs", help="Output directory for logs/checkpoints")
    parser.add_argument("--config", default=None, help="Optional JSON config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt = train(args.data, args.out_dir, args.config)
    log.info("saved_checkpoint=%s", ckpt)


if __name__ == "__main__":
    main()
