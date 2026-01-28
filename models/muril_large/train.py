"""
Training script for MuRIL-Large on POCSO Legal Dialogue Dataset
MuRIL is optimized for Hindi-English code-mixed text
"""

import os
import json
import yaml
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def prepare_data(entry, tokenizer, max_length):
    """Prepare data for classification task"""
    turns = entry.get('turns', [])
    user_texts = [turn['text'] for turn in turns if turn['role'] == 'user']
    input_text = " ".join(user_texts)
    
    # Use complexity as label
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
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    print("="*70)
    print("Training MuRIL-Large on POCSO Legal Dialogue Dataset")
    print("="*70)
    
    print(f"\n1. Loading model: {config['model']['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['model_name'],
        num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'])
    
    print("\n2. Loading training data...")
    train_path = os.path.join('experiments', config['data']['exp1_train'])
    train_data = load_jsonl(train_path)
    print(f"   Loaded {len(train_data)} training examples")
    
    print("\n3. Preparing dataset...")
    train_dataset = []
    for entry in tqdm(train_data[:100]):  # Limit for testing
        prepared = prepare_data(entry, tokenizer, config['training']['max_length'])
        train_dataset.append(prepared)
    
    train_dataset = Dataset.from_list(train_dataset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=config['output']['checkpoint_dir'],
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
    final_model_path = os.path.join(config['output']['checkpoint_dir'], 'final')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("\n" + "="*70)
    print("✅ Training completed!")
    print(f"Model saved to: {final_model_path}")
    print("="*70)

if __name__ == '__main__':
    main()
