"""SFT examples with prompt tokens masked in labels (-100)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from research_pipeline.utils import format_dialogue_prompt, prompt_prefix_tokens_len


class LegalSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        return_row_index: bool = False,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_row_index = return_row_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        user = row.get("input", "").strip()
        assistant = row.get("output", "").strip()
        text = format_dialogue_prompt(user, assistant)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        plen = min(prompt_prefix_tokens_len(self.tokenizer, user), len(input_ids))
        labels = input_ids.copy()
        for i in range(plen):
            labels[i] = -100
        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
        if self.return_row_index:
            out["_row_index"] = idx
        return out


def collate_sft_batch(batch: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor | list[int]]:
    """Pad sequences; mask pad tokens in labels."""
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attn = []
    labels = []
    indices: list[int] = []
    for x in batch:
        ids = x["input_ids"]
        m = x["attention_mask"]
        lab = x["labels"]
        extra = max_len - len(ids)
        input_ids.append(ids + [pad_id] * extra)
        attn.append(m + [0] * extra)
        labels.append(lab + [-100] * extra)
        if "_row_index" in x:
            indices.append(int(x["_row_index"]))
    batch_tensors: dict[str, torch.Tensor | list[int]] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if indices:
        batch_tensors["_row_indices"] = indices
    return batch_tensors
