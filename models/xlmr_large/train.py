"""
Training script for XLM-RoBERTa-Large on POCSO Legal Dialogue Dataset.
Exp1: Fine-tuning only on dialogue (classification). Exp3: Load pretrained (MLM) then fine-tune.
"""

import os
import sys
import json
import yaml
import argparse
import torch
from transformers import (
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl(filepath):
    data = []
    path = filepath if os.path.isabs(filepath) else os.path.join(ROOT, filepath)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def prepare_data(entry, tokenizer, max_length):
    """Prepare data for classification (complexity)."""
    turns = entry.get('turns', [])
    user_texts = [t['text'] for t in turns if t.get('role') == 'user']
    input_text = " ".join(user_texts) if user_texts else entry.get('input', '')
    label = {'layman': 0, 'intermediate': 1, 'professional': 2}[entry.get('complexity', 'layman')]
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    encoding['labels'] = label
    return encoding


def main():
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa (Exp1, Exp2, or Exp3)')
    parser.add_argument('--experiment', type=str, default='exp1', choices=['exp1', 'exp2', 'exp3'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Exp2/Exp3: path to MLM-pretrained checkpoint')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    xlmr_dir = os.path.dirname(__file__)

    if args.experiment == 'exp1':
        train_path = os.path.join('experiments', config['data']['exp1_train'])
        output_subdir = 'exp1'
        pretrained_path = None
        freeze_encoder = False
    elif args.experiment == 'exp2':
        train_path = os.path.join('experiments', config['data']['exp1_train'])
        output_subdir = 'exp2'
        pretrained_path = args.checkpoint or os.path.join(xlmr_dir, 'checkpoints/exp2/pretrained/final')
        freeze_encoder = True  # Exp2: pretrain only -> train classification head only
    else:
        train_path = os.path.join(ROOT, 'experiments/exp3_pretraining_finetuning/finetuning/train.jsonl')
        output_subdir = 'exp3'
        pretrained_path = args.checkpoint or os.path.join(xlmr_dir, 'checkpoints/exp3/pretrained/final')
        freeze_encoder = False

    print("="*70)
    print("Training XLM-RoBERTa-Large on POCSO Legal Dialogue Dataset")
    print(f"Experiment: {args.experiment}")
    print("="*70)

    print(f"\n1. Loading tokenizer and model...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['model']['tokenizer_name'])
    if (args.experiment in ('exp2', 'exp3')) and pretrained_path and os.path.exists(pretrained_path):
        from transformers import XLMRobertaForMaskedLM
        mlm_model = XLMRobertaForMaskedLM.from_pretrained(pretrained_path)
        model = XLMRobertaForSequenceClassification.from_pretrained(
            config['model']['model_name'],
            num_labels=3
        )
        model.roberta.load_state_dict(mlm_model.roberta.state_dict(), strict=True)
        del mlm_model
        print(f"   Loaded encoder from {pretrained_path}")
        if freeze_encoder:
            for p in model.roberta.parameters():
                p.requires_grad = False
            print("   Frozen encoder (Exp2: training classification head only)")
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(
            config['model']['model_name'],
            num_labels=3
        )

    print("\n2. Loading training data...")
    train_path_abs = train_path if os.path.isabs(train_path) else os.path.join(ROOT, train_path)
    train_data = load_jsonl(train_path_abs)
    if not train_data and not os.path.isabs(train_path):
        train_data = load_jsonl(train_path)
    print(f"   Loaded {len(train_data)} training examples")
    
    print("\n3. Preparing dataset...")
    train_dataset = []
    for entry in tqdm(train_data, desc="Preparing"):
        prepared = prepare_data(entry, tokenizer, config['training']['max_length'])
        train_dataset.append(prepared)
    
    train_dataset = Dataset.from_list(train_dataset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    checkpoint_dir = config['output']['checkpoint_dir']
    exp_dir = os.path.join(checkpoint_dir, output_subdir)
    os.makedirs(exp_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=exp_dir,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        logging_dir=config['output']['log_dir'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        seed=config['training']['seed']
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("\n4. Starting training...")
    trainer.train()
    
    print("\n5. Saving final model...")
    final_model_path = os.path.join(config['output']['checkpoint_dir'], output_subdir, 'final')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("\n" + "="*70)
    print("✅ Training completed!")
    print(f"Model saved to: {final_model_path}")
    print("="*70)

if __name__ == '__main__':
    main()
