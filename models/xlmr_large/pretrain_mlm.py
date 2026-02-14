"""
MLM pretraining for XLM-RoBERTa on legal corpus (Exp2 and Exp3).
Saves encoder to checkpoints/exp2/pretrained/final or checkpoints/exp3/pretrained/final.
"""

import os
import sys
import yaml
import argparse
import torch
from transformers import (
    XLMRobertaForMaskedLM,
    XLMRobertaTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_corpus_text(data_dir, max_chars=None):
    texts = []
    if os.path.isfile(data_dir) and data_dir.endswith('.txt'):
        with open(data_dir, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    elif os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith('.txt'):
                with open(os.path.join(data_dir, name), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    full = "\n\n".join(texts)
    if max_chars:
        full = full[:max_chars]
    return full


def tokenize_and_chunk(text, tokenizer, block_size=512):
    step = 50000
    all_ids = []
    for i in range(0, len(text), step):
        segment = text[i:i + step]
        enc = tokenizer(segment, return_tensors=None, add_special_tokens=True,
                       truncation=True, max_length=block_size * 200)
        all_ids.extend(enc['input_ids'])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    blocks = []
    for i in range(0, len(all_ids), block_size):
        block = all_ids[i:i + block_size]
        real_len = len(block)
        if real_len < block_size:
            block = block + [pad_id] * (block_size - real_len)
        mask = [1] * real_len + [0] * (block_size - real_len)
        blocks.append({'input_ids': block, 'attention_mask': mask})
    return blocks


def main():
    parser = argparse.ArgumentParser(description='MLM pretrain XLM-RoBERTa on legal corpus')
    parser.add_argument('--experiment', type=str, default='exp2', choices=['exp2', 'exp3'])
    parser.add_argument('--max_chars', type=int, default=None, help='Cap corpus size (chars) for debugging')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    model_name = config['model']['model_name']

    if args.experiment == 'exp2':
        data_dir = os.path.join(ROOT, 'experiments/exp2_pretraining_only/pretraining/legal_corpus')
    else:
        data_dir = os.path.join(ROOT, 'experiments/exp3_pretraining_finetuning/pretraining/legal_corpus')

    if not os.path.exists(data_dir):
        print(f"Data dir not found: {data_dir}")
        return

    print("="*70)
    print("XLM-RoBERTa MLM Pretraining on Legal Corpus")
    print(f"Experiment: {args.experiment}")
    print("="*70)

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    text = load_corpus_text(data_dir, max_chars=args.max_chars)
    print(f"Corpus size: {len(text)} chars")
    blocks = tokenize_and_chunk(text, tokenizer, block_size=config['training'].get('max_length', 512))
    train_dataset = Dataset.from_list(blocks)
    print(f"Training blocks: {len(train_dataset)}")

    model = XLMRobertaForMaskedLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    out_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', args.experiment, 'pretrained')
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config['training'].get('pretrain_epochs', 2),
        per_device_train_batch_size=config['training'].get('batch_size', 4),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 4),
        learning_rate=float(config['training'].get('pretrain_lr', config['training']['learning_rate'])),
        warmup_steps=config['training'].get('warmup_steps', 500),
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=config['training'].get('fp16', True),
        seed=config['training'].get('seed', 42),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    final_dir = os.path.join(out_dir, 'final')
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✅ Pretraining done. Saved to {final_dir}")


if __name__ == '__main__':
    main()
