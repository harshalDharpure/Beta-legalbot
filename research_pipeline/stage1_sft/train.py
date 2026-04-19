#!/usr/bin/env python3
"""
Stage 1: supervised fine-tuning (causal LM CE on assistant tokens only).

  PYTHONPATH=. python research_pipeline/stage1_sft/train.py \\
    --config research_pipeline/configs/pipeline_default.yaml \\
    --train-jsonl research_pipeline/data/splits/train.jsonl \\
    --val-jsonl research_pipeline/data/splits/val.jsonl \\
    --output-dir research_pipeline/checkpoints/stage1/M1_seed42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research_pipeline.stage1_sft.dataset import LegalSFTDataset, collate_sft_batch
from research_pipeline.utils import load_jsonl, set_global_seed


def load_yaml(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    cfg_path = args.config if os.path.isabs(args.config) else str(_REPO / args.config)
    cfg = load_yaml(cfg_path)
    base = cfg["project"]["base_model"]
    tc = cfg.get("training", {})

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    train_path = args.train_jsonl if os.path.isabs(args.train_jsonl) else str(_REPO / args.train_jsonl)
    val_path = args.val_jsonl if args.val_jsonl and os.path.isabs(args.val_jsonl) else (str(_REPO / args.val_jsonl) if args.val_jsonl else "")
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path and os.path.isfile(val_path) else None

    max_len = int(tc.get("max_length", 512))
    train_ds = LegalSFTDataset(train_rows, tokenizer, max_len)
    val_ds = LegalSFTDataset(val_rows, tokenizer, max_len) if val_rows else None

    collate_fn = lambda b: collate_sft_batch(b, tokenizer)

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else str(_REPO / args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    ta_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(tc.get("per_device_batch_size", 2)),
        gradient_accumulation_steps=int(tc.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tc.get("learning_rate", 5e-5)),
        num_train_epochs=float(tc.get("num_epochs", 3)),
        logging_steps=int(tc.get("logging_steps", 50)),
        save_steps=int(tc.get("save_steps", 500)),
        save_total_limit=int(tc.get("save_total_limit", 3)),
        fp16=bool(tc.get("fp16", True)),
        report_to="none",
        seed=args.seed,
    )
    if val_ds:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["eval_steps"] = int(tc.get("eval_steps", 500))
        ta_kwargs["load_best_model_at_end"] = True
        ta_kwargs["metric_for_best_model"] = "loss"
    else:
        ta_kwargs["eval_strategy"] = "no"
    args_tr = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))
    print("Saved", os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()
