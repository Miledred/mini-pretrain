#!/usr/bin/env python3
"""
Generate fixed-prompt samples from a trained checkpoint and save them to Markdown.

Run from the mini_pretrain project root:
    python tools/phase1_sample_checkpoints.py \
        --checkpoint checkpoints/my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel/best.pt

This script assumes the project has a src/ package containing config.py,
model_factory.py, and tokenizer.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_PROMPTS = [
    "The history of science",
    "In the early twentieth century",
    "Artificial intelligence is",
    "The theory of evolution states that",
    "In modern society, technology has",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="phase1_outputs/sample_outputs.md")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def load_cfg_from_checkpoint(ckpt_cfg: dict):
    from src.config import TrainConfig

    cfg = TrainConfig()
    for key, value in ckpt_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_model(checkpoint: Path):
    from src.model_factory import build_model
    from src.tokenizer import build_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = load_cfg_from_checkpoint(ckpt.get("cfg", {}))
    tokenizer = build_tokenizer(cfg.tokenizer_name)
    model = build_model(cfg, tokenizer)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return cfg, tokenizer, model, device, ckpt


@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, top_k: int, top_p: float):
    enc = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_k": top_k, "top_p": top_p})

    out = model.generate(**kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True), input_ids.shape[1], out.shape[1]


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))

    checkpoint = (project_root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    out = (project_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg, tokenizer, model, device, ckpt = load_model(checkpoint)
    lines: list[str] = []
    lines.append("# Phase 1 Fixed-Prompt Sampling")
    lines.append("")
    lines.append(f"- Checkpoint: `{checkpoint}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Loaded step: `{ckpt.get('step', 'N/A')}`")
    lines.append(f"- Best val loss: `{ckpt.get('best_val_loss', 'N/A')}`")
    lines.append(f"- Model: L{cfg.n_layer}-H{cfg.n_head}-D{cfg.n_embd}, block_size={cfg.block_size}")
    lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    lines.append("")

    for idx, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        prompt_ids = tokenizer.encode(prompt)
        total_len = len(prompt_ids) + args.max_new_tokens
        lines.append(f"## Prompt {idx}: {prompt}")
        lines.append("")
        lines.append(f"- prompt_tokens: `{len(prompt_ids)}`")
        lines.append(f"- prompt_tokens + max_new_tokens: `{total_len}`")
        if total_len > cfg.block_size:
            lines.append(f"- skipped: total length exceeds block_size `{cfg.block_size}`")
            lines.append("")
            continue

        greedy, greedy_in, greedy_out = generate(
            model, tokenizer, device, prompt, args.max_new_tokens, False, args.temperature, args.top_k, args.top_p
        )
        sampled, sampled_in, sampled_out = generate(
            model, tokenizer, device, prompt, args.max_new_tokens, True, args.temperature, args.top_k, args.top_p
        )

        lines.append("### Greedy")
        lines.append("")
        lines.append(f"input_len={greedy_in}, output_len={greedy_out}")
        lines.append("")
        lines.append("```text")
        lines.append(greedy)
        lines.append("```")
        lines.append("")
        lines.append("### Sampling")
        lines.append("")
        lines.append(f"input_len={sampled_in}, output_len={sampled_out}, temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        lines.append("")
        lines.append("```text")
        lines.append(sampled)
        lines.append("```")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Sample report written to: {out}")


if __name__ == "__main__":
    main()
