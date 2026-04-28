from array import array
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def clean_lines(lines):
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("=") and line.endswith("="):
            continue

        cleaned.append(line)
    return cleaned


def choose_storage_type(vocab_size: int):
    if vocab_size <= 65535:
        return "H", "uint16"
    if vocab_size <= 4294967295:
        return "I", "uint32"
    raise ValueError(f"vocab_size too large: {vocab_size}")


def write_token_file(lines, tokenizer, out_path: Path, chunk_size: int = 2000):
    type_code, dtype_name = choose_storage_type(len(tokenizer))
    eos_id = tokenizer.eos_token_id
    total_tokens = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        for start in range(0, len(lines), chunk_size):
            batch_lines = lines[start:start + chunk_size]

            encoded_batch = tokenizer(
                batch_lines,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            buf = array(type_code)
            for ids in encoded_batch:
                buf.extend(ids)
                if eos_id is not None:
                    buf.append(eos_id)

            buf.tofile(f)
            total_tokens += len(buf)

            end = min(start + chunk_size, len(lines))
            print(f"{out_path.name}: {end}/{len(lines)} lines, {total_tokens} tokens")

    return {
        "file": out_path.name,
        "dtype": dtype_name,
        "num_tokens": total_tokens,
    }


def main():
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    tokenizer_name = "gpt2"
    dataset_name = "wikitext-103-raw-v1"

    print(f"Loading {dataset_name} ...")
    ds = load_dataset("Salesforce/wikitext", dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_lines = clean_lines(ds["train"]["text"])
    val_lines = clean_lines(ds["validation"]["text"])

    train_text = "\n\n".join(train_lines)
    val_text = "\n\n".join(val_lines)

    train_txt_path = data_dir / "train.txt"
    val_txt_path = data_dir / "val.txt"

    train_txt_path.write_text(train_text, encoding="utf-8")
    val_txt_path.write_text(val_text, encoding="utf-8")

    train_meta = write_token_file(train_lines, tokenizer, data_dir / "train.bin")
    val_meta = write_token_file(val_lines, tokenizer, data_dir / "val.bin")

    meta = {
        "tokenizer_name": tokenizer_name,
        "vocab_size": len(tokenizer),
        "train": train_meta,
        "val": val_meta,
    }

    meta_path = data_dir / "data_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train text written to:   {train_txt_path}")
    print(f"val text written to:     {val_txt_path}")
    print(f"train tokens written to: {data_dir / 'train.bin'}")
    print(f"val tokens written to:   {data_dir / 'val.bin'}")
    print(f"meta written to:         {meta_path}")


if __name__ == "__main__":
    main()