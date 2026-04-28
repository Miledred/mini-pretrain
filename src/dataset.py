import json

import numpy as np
import torch
from torch.utils.data import Dataset


DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


def load_data_meta(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MemMapTokenBlockDataset(Dataset):
    def __init__(self, token_file: str, dtype_name: str, block_size: int, stride: int | None = None):
        self.token_file = token_file
        self.block_size = block_size
        self.stride = block_size if stride is None else stride

        if dtype_name not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype_name: {dtype_name}")

        self.tokens = np.memmap(
            self.token_file,
            mode="r",
            dtype=DTYPE_MAP[dtype_name],
        )
        self.num_tokens = int(self.tokens.shape[0])

        self.max_start = self.num_tokens - (self.block_size + 1)
        if self.max_start < 0:
            self.num_examples = 0
        else:
            self.num_examples = self.max_start // self.stride + 1

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.block_size + 1

        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)

        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())

        return {
            "input_ids": input_ids,
            "labels": labels,
        }