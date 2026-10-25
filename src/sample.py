import argparse
import os
import sys

import torch

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.config import TrainConfig
from src.model_factory import build_model
from src.tokenizer import build_tokenizer


def load_cfg_from_checkpoint(ckpt_cfg: dict) -> TrainConfig:
    cfg = TrainConfig()
    for key, value in ckpt_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_model_from_checkpoint(checkpoint_path: str, device_arg: str | None = None):
    if device_arg is not None:
        device = torch.device(device_arg if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    ckpt_cfg = ckpt.get("cfg", {})
    cfg = load_cfg_from_checkpoint(ckpt_cfg)

    tokenizer = build_tokenizer(cfg.tokenizer_name)
    model = build_model(cfg, tokenizer)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    step = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", None)

    return cfg, tokenizer, model, device, step, best_val_loss


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 48,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_k"] = top_k
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["temperature"] = 1.0

    output_ids = model.generate(**generate_kwargs)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, input_ids.shape[1], output_ids.shape[1]


def print_generation_block(title: str, text: str):
    print(f"[{title}]")
    print(text)
    print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint file, such as checkpoints/.../best.pt",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt for generation. Can be used multiple times.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    fixed_prompts = [
        "The history of science",
        "In the early twentieth century",
        "Artificial intelligence is",
        "The theory of evolution states that",
        "In modern society, technology has",
    ]

    prompts = args.prompt if args.prompt is not None else fixed_prompts

    cfg, tokenizer, model, device, step, best_val_loss = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device_arg=args.device,
    )

    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Loaded step: {step}")
    print(f"Best val loss: {best_val_loss}")
    print(f"Block size: {cfg.block_size}")
    print(f"Layers: {cfg.n_layer}, Heads: {cfg.n_head}, Embedding dim: {cfg.n_embd}")
    print("=" * 80)

    for idx, prompt in enumerate(prompts, start=1):
        prompt_ids = tokenizer.encode(prompt)
        total_len = len(prompt_ids) + args.max_new_tokens

        print(f"\n[Prompt {idx}]")
        print(f"Prompt: {prompt}")
        print(f"Prompt tokens: {len(prompt_ids)}")
        print(f"Prompt tokens + max_new_tokens: {total_len}")

        if total_len > cfg.block_size:
            print(
                f"Skip: total length {total_len} exceeds block_size {cfg.block_size}. "
                f"Use a shorter prompt or smaller max_new_tokens."
            )
            print("-" * 80)
            continue

        greedy_text, input_len_greedy, output_len_greedy = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        sampled_text, input_len_sampled, output_len_sampled = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print(f"Greedy input length: {input_len_greedy}")
        print(f"Greedy output length: {output_len_greedy}")
        print_generation_block("Greedy", greedy_text)

        print(f"Sampled input length: {input_len_sampled}")
        print(f"Sampled output length: {output_len_sampled}")
        print_generation_block("Sampled", sampled_text)


if __name__ == "__main__":
    main()