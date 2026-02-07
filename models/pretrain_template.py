"""
Pretraining script for Exp2 and Exp3.
Performs Causal Language Modeling (CLM) on legal corpus.
"""

import os
import json
import yaml
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_text(data_dir):
    """Load all .txt content from a file or directory."""
    texts = []
    if os.path.isfile(data_dir) and data_dir.endswith('.txt'):
        with open(data_dir, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    elif os.path.isdir(data_dir):
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    return texts


def prepare_pretraining_data(data_dir, tokenizer, max_length=512):
    """Prepare pretraining data: tokenize and split into fixed-length blocks."""
    all_text = load_all_text(data_dir)
    if not all_text:
        return Dataset.from_list([])
    # Tokenize all text and chunk into blocks of max_length
    all_ids = []
    for text in tqdm(all_text, desc="Tokenizing"):
        # Tokenize in chunks to avoid huge strings (max ~100k tokens per file segment)
        step = 50000  # chars per chunk
        for i in range(0, len(text), step):
            segment = text[i:i + step]
            enc = tokenizer(segment, return_tensors=None, add_special_tokens=True, truncation=True, max_length=512 * 200)
            all_ids.extend(enc['input_ids'])
    # Split into blocks
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    blocks = []
    for i in range(0, len(all_ids), max_length):
        block = all_ids[i:i + max_length]
        real_len = len(block)
        if real_len < max_length:
            block = block + [pad_id] * (max_length - real_len)
        mask = [1] * real_len + [0] * (max_length - real_len)
        blocks.append({'input_ids': block, 'attention_mask': mask})
    return Dataset.from_list(blocks)


def setup_model_and_tokenizer(config, use_qlora=False, gpu_ids=None):
    """Setup model and tokenizer for pretraining."""
    model_name = config['model']['model_name']
    tokenizer_name = config['model'].get('tokenizer_name', model_name)
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {model_name}")
    use_4bit = use_qlora and config['model'].get('quantization') == '4bit' and BNB_AVAILABLE
    if use_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config['training'].get('fp16', True) else torch.float32,
            device_map='auto' if (gpu_ids is None or use_qlora) else None
        )
        if gpu_ids is not None and not use_qlora:
            if len(gpu_ids) == 1:
                model = model.to(f'cuda:{gpu_ids[0]}')
            else:
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
    # Setup QLoRA if needed
    if use_qlora:
        print("Setting up QLoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def pretrain_model(model_name, experiment='exp2', gpu_ids=None, use_multi_gpu=True):
    """Pretrain a model on legal corpus."""
    print(f"\n{'='*70}")
    print(f"Pretraining {model_name} for {experiment}")
    print(f"{'='*70}")
    
    # Load config
    model_dir = os.path.join('models', model_name)
    config_path = os.path.join(model_dir, 'config.yaml')
    config = load_config(config_path)
    
    # Setup model
    use_qlora = config['model'].get('use_qlora', False)
    model, tokenizer = setup_model_and_tokenizer(config, use_qlora=use_qlora, gpu_ids=gpu_ids)
    
    # Reduce memory for full-model pretraining
    if not use_qlora and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Load pretraining data
    if experiment == 'exp2':
        data_dir = 'experiments/exp2_pretraining_only/pretraining/legal_corpus'
    elif experiment == 'exp3':
        data_dir = 'experiments/exp3_pretraining_finetuning/pretraining/legal_corpus'
    else:
        data_dir = f'experiments/{experiment}/pretraining/legal_corpus'
    
    if not os.path.exists(data_dir):
        print(f"❌ Pretraining data directory not found: {data_dir}")
        print("Please prepare legal corpus data first.")
        return False
    
    print(f"\nLoading pretraining data from {data_dir}...")
    train_dataset = prepare_pretraining_data(
        data_dir, 
        tokenizer,
        max_length=config['training'].get('max_length', 512)
    )
    print(f"Pretraining samples: {len(train_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    if experiment == 'exp2':
        output_dir = os.path.join(model_dir, 'checkpoints', 'exp2', 'pretrained')
    elif experiment == 'exp3':
        output_dir = os.path.join(model_dir, 'checkpoints', 'exp3', 'pretrained')
    else:
        output_dir = os.path.join(model_dir, 'checkpoints', experiment, 'pretrained')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of GPUs
    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = len(gpu_ids) if isinstance(gpu_ids, list) else 1
    
    per_device_batch = config['training'].get('pretrain_batch_size', config['training'].get('batch_size', 2))
    if num_gpus > 1:
        per_device_batch = max(1, per_device_batch // 2)
    grad_accum = config['training'].get('pretrain_gradient_accumulation', config['training'].get('gradient_accumulation_steps', 8))
    
    use_fp16 = config['training'].get('fp16', True) and num_gpus == 1
    use_bf16 = config['training'].get('bf16', (not use_fp16) and num_gpus == 1)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training'].get('pretrain_epochs', 2),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(config['training'].get('pretrain_lr', config['training']['learning_rate'])),
        warmup_steps=config['training'].get('pretrain_warmup_steps', config['training']['warmup_steps']),
        logging_dir=os.path.join(model_dir, 'logs', experiment, 'pretraining'),
        logging_steps=config['training'].get('logging_steps', 50),
        save_steps=config['training'].get('pretrain_save_steps', 500),
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 0),
        seed=config['training'].get('seed', 42),
        report_to=[],
        dataloader_pin_memory=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Train
    print(f"\n🚀 Starting pretraining...")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {per_device_batch} × {num_gpus} GPUs × {training_args.gradient_accumulation_steps} accumulation")
    print(f"   Total steps: ~{len(train_dataset) // (per_device_batch * num_gpus * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    trainer.train()
    
    # Save final model
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    
    if use_qlora:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    else:
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
    
    print(f"\n✅ Pretraining complete! Model saved to {final_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Pretrain models on legal corpus')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--experiment', type=str, default='exp2', choices=['exp2', 'exp3'], help='Experiment')
    parser.add_argument('--gpu', type=int, nargs='+', default=None, help='GPU IDs (e.g., --gpu 0 1)')
    
    args = parser.parse_args()
    
    gpu_ids = args.gpu if args.gpu else None
    pretrain_model(args.model, args.experiment, gpu_ids=gpu_ids)


if __name__ == '__main__':
    main()
