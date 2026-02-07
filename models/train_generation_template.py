"""
Template training script for generation models (Exp1: Finetuning only).
Supports both QLoRA (for large models) and full fine-tuning (for small models).
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from tqdm import tqdm


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def format_prompt(input_text, output_text=None):
    """Format prompt for instruction-following models."""
    prompt = f"User: {input_text}\nAssistant:"
    if output_text:
        prompt += f" {output_text}"
    return prompt


def prepare_dataset(data, tokenizer, max_length=512, max_target_length=256):
    """Prepare dataset for generation task."""
    dataset = []
    for entry in tqdm(data, desc="Preparing dataset"):
        input_text = entry.get('input', '')
        output_text = entry.get('output', '')
        
        # Format prompt
        prompt = format_prompt(input_text, output_text)
        
        # Tokenize
        encoding = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For generation, labels are the same as input_ids (shifted)
        encoding['labels'] = encoding['input_ids'].clone()
        
        dataset.append({
            'input_ids': encoding['input_ids'].squeeze().tolist(),
            'attention_mask': encoding['attention_mask'].squeeze().tolist(),
            'labels': encoding['labels'].squeeze().tolist()
        })
    
    return Dataset.from_list(dataset)


def setup_model_and_tokenizer(config, use_qlora=False, gpu_ids=None):
    """Setup model and tokenizer with optional QLoRA."""
    model_name = config['model']['model_name']
    tokenizer_name = config['model'].get('tokenizer_name', model_name)
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {model_name}")
    # For multi-GPU, don't use device_map='auto' - let DataParallel handle it
    device_map = None if use_qlora or (gpu_ids and len(gpu_ids) > 1) else 'auto'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config['training'].get('fp16', True) else torch.float32,
        device_map=device_map
    )
    
    # Move to first GPU if not using device_map
    if device_map is None:
        model = model.to('cuda:0')
    
    # Setup QLoRA if needed
    if use_qlora:
        print("Setting up QLoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_model_and_tokenizer_from_exp2(model_name, config, use_qlora=False, gpu_ids=None):
    """Load model and tokenizer from Exp2 checkpoint (for Exp3: continue finetuning on dialogue)."""
    tokenizer_name = config['model'].get('tokenizer_name', config['model']['model_name'])
    exp2_final = os.path.join('models', model_name, 'checkpoints', 'exp2', 'pretrained', 'final')
    if not os.path.isdir(exp2_final):
        raise FileNotFoundError(f"Exp2 checkpoint not found: {exp2_final}")
    has_adapter = os.path.isfile(os.path.join(exp2_final, 'adapter_config.json'))
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if config['training'].get('fp16', True) else torch.float32
    # Use single device to avoid device index mismatch (e.g. when CUDA_VISIBLE_DEVICES=0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if has_adapter:
        print(f"Loading base model then Exp2 adapter from {exp2_final}")
        base = AutoModelForCausalLM.from_pretrained(
            config['model']['model_name'],
            torch_dtype=dtype,
            device_map={'': 0} if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available():
            base = base.to(device)
        model = PeftModel.from_pretrained(base, exp2_final)
        model = model.merge_and_unload()
        model = model.to(device)
    else:
        print(f"Loading full model from Exp2: {exp2_final}")
        model = AutoModelForCausalLM.from_pretrained(exp2_final, torch_dtype=dtype, device_map={'': 0} if torch.cuda.is_available() else None)
        if not torch.cuda.is_available():
            model = model.to(device)
    if use_qlora:
        print("Setting up QLoRA for Exp3 finetuning...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model, tokenizer


def train_model(model_name, experiment='exp1', gpu_ids=None, use_multi_gpu=True):
    """Train a model for generation task with multi-GPU support."""
    model_dir = os.path.join('models', model_name)
    config_path = os.path.join(model_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    
    config = load_config(config_path)
    
    # Determine GPU IDs
    if gpu_ids is None:
        # Find free GPUs
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        free_gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_free = int(parts[1])
                utilization = int(parts[2])
                if memory_free > 30000 and utilization < 10:  # >30GB free, <10% utilization
                    free_gpus.append(gpu_id)
        
        if not free_gpus:
            print("❌ No free GPUs found!")
            return False
        
        # Use up to 4 GPUs for multi-GPU training
        gpu_ids = free_gpus[:4] if use_multi_gpu and len(free_gpus) > 1 else [free_gpus[0]]
    
    # Set GPU visibility - Trainer will automatically use DataParallel
    gpu_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPU(s): {gpu_ids}")
    
    # Move model to device (Trainer will handle multi-GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"Training {model_name} on {experiment}")
    print(f"GPUs: {gpu_ids} ({num_gpus} GPU(s))")
    print("="*70)
    
    # Setup model (Exp3: load from Exp2 checkpoint; Exp1: load from base)
    use_qlora = config['model'].get('use_qlora', False)
    if experiment == 'exp3':
        model, tokenizer = setup_model_and_tokenizer_from_exp2(model_name, config, use_qlora=use_qlora, gpu_ids=gpu_ids)
    else:
        model, tokenizer = setup_model_and_tokenizer(config, use_qlora=use_qlora, gpu_ids=gpu_ids)
    
    # Load data
    if experiment == 'exp1':
        train_path = 'experiments/exp1_finetuning_only/data/train_70.jsonl'
        val_path = 'experiments/exp1_finetuning_only/data/val_10.jsonl'
    elif experiment == 'exp3':
        train_path = 'experiments/exp3_pretraining_finetuning/finetuning/train.jsonl'
        val_path = 'experiments/exp3_pretraining_finetuning/finetuning/val.jsonl'
    else:
        print(f"❌ Unknown experiment: {experiment}")
        return False
    
    print(f"\nLoading data from {train_path}...")
    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    
    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, 
                                     max_length=config['training']['max_length'],
                                     max_target_length=config['training']['max_target_length'])
    val_dataset = prepare_dataset(val_data, tokenizer,
                                  max_length=config['training']['max_length'],
                                  max_target_length=config['training']['max_target_length'])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    output_dir = os.path.join(model_dir, 'checkpoints', experiment)
    os.makedirs(output_dir, exist_ok=True)
    
    # Adjust batch size for multi-GPU and for Exp3 / full fine-tuning (memory-heavy)
    per_device_batch = config['training']['batch_size']
    if num_gpus > 1:
        per_device_batch = max(1, per_device_batch // 2)
        print(f"Multi-GPU: Adjusted per-device batch size to {per_device_batch}")
    if experiment == 'exp3' or not use_qlora:
        per_device_batch = min(per_device_batch, 2)  # Exp3 / full FT often OOM
        if experiment == 'exp3':
            per_device_batch = 1
        print(f"Exp3/full FT: per_device_batch={per_device_batch}")
    
    # Gradient checkpointing for Exp3 / full model to save memory
    if experiment == 'exp3' or not use_qlora:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    
    # Disable FP16 for DataParallel (it has issues with FP16)
    use_fp16 = config['training'].get('fp16', True) and num_gpus == 1
    use_bf16 = config['training'].get('bf16', False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=config['training']['warmup_steps'],
        logging_dir=os.path.join(model_dir, 'logs', experiment),
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training'].get('eval_steps', 500),
        eval_strategy='steps',
        save_total_limit=config['training']['save_total_limit'],
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 4),
        seed=config['training']['seed'],
        report_to=[],  # Disable wandb/tensorboard
        dataloader_pin_memory=True  # Faster data loading
    )
    
    if num_gpus > 1 and not use_fp16:
        print(f"⚠️ FP16 disabled for multi-GPU training (DataParallel compatibility)")
    
    # Wrap model with DataParallel for multi-GPU (only if model fits)
    # For models with memory issues, use single GPU or reduce batch size
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        # Check model size - if too large, use single GPU
        try:
            print(f"✅ Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        except Exception as e:
            print(f"⚠️ DataParallel failed: {e}")
            print(f"   Falling back to single GPU (GPU 0)")
            num_gpus = 1
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0]) if gpu_ids else '0'
    
    # Trainer
    trainer = Trainer(
        model=model.module if isinstance(model, torch.nn.DataParallel) else model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    final_dir = os.path.join(output_dir, 'final')
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n✅ Training complete! Model saved to {final_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., qwen2.5_1.5b)')
    parser.add_argument('--experiment', type=str, default='exp1', choices=['exp1', 'exp3'], help='Experiment name')
    parser.add_argument('--gpu', type=int, nargs='+', default=None, help='GPU ID(s) - can specify multiple: --gpu 0 1 4')
    parser.add_argument('--multi-gpu', action='store_true', default=True, help='Use multiple GPUs if available')
    
    args = parser.parse_args()
    train_model(args.model, args.experiment, gpu_ids=args.gpu, use_multi_gpu=args.multi_gpu)
