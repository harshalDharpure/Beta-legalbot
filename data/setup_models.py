"""
Setup model folders for all 7 models.
"""

import os
from pathlib import Path


def create_model_structure(model_name: str, model_info: dict):
    """Create folder structure for a model."""
    model_dir = Path('models') / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (model_dir / 'checkpoints').mkdir(exist_ok=True)
    (model_dir / 'results').mkdir(exist_ok=True)
    (model_dir / 'logs').mkdir(exist_ok=True)
    
    # Create config.yaml
    config_path = model_dir / 'config.yaml'
    if not config_path.exists():
        config_content = f"""# Configuration for {model_info['display_name']}

model:
  model_name: "{model_info['model_name']}"
  tokenizer_name: "{model_info.get('tokenizer_name', model_info['model_name'])}"
  use_qlora: {model_info.get('use_qlora', False)}
  quantization: "{model_info.get('quantization', 'none')}"

training:
  batch_size: {model_info.get('batch_size', 4)}
  gradient_accumulation_steps: {model_info.get('gradient_accumulation', 4)}
  learning_rate: {model_info.get('learning_rate', 5e-5)}
  num_epochs: {model_info.get('num_epochs', 10)}
  warmup_steps: 500
  max_length: 512
  max_target_length: 256
  save_steps: 500
  eval_steps: 500
  logging_steps: 100
  save_total_limit: 3
  fp16: true
  dataloader_num_workers: 4
  seed: 42

data:
  train_path: "experiments/exp1_finetuning_only/data/train_70.jsonl"
  val_path: "experiments/exp1_finetuning_only/data/val_10.jsonl"
  test_path: "experiments/exp1_finetuning_only/data/test_20.jsonl"

output:
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  results_dir: "results"
"""
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ Created config.yaml for {model_name}")
    
    # Create README.md
    readme_path = model_dir / 'README.md'
    if not readme_path.exists():
        readme_content = f"""# {model_info['display_name']}

## Model Information

- **Model Name**: {model_info['model_name']}
- **Type**: {model_info.get('type', 'Generation Model')}
- **Quantization**: {model_info.get('quantization', 'None')}
- **QLoRA**: {model_info.get('use_qlora', False)}

## Training

### Exp1: Finetuning Only
```bash
python train.py --experiment exp1
```

### Exp2: Pretraining Only
```bash
python pretrain.py --experiment exp2
python evaluate.py --experiment exp2 --checkpoint checkpoints/exp2_pretraining/
```

### Exp3: Pretraining + Finetuning
```bash
python pretrain.py --experiment exp3
python train.py --experiment exp3 --checkpoint checkpoints/exp3_pretraining/
```

## Results

Results are saved in `results/` directory:
- `exp1_results.json` - Exp1 results
- `exp2_results.json` - Exp2 results
- `exp3_results.json` - Exp3 results
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ Created README.md for {model_name}")


def main():
    """Create model folders."""
    models = {
        'llama3.1_8b': {
            'display_name': 'LLaMA-3.1-8B',
            'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'use_qlora': True,
            'quantization': '4bit',
            'batch_size': 2,
            'gradient_accumulation': 8
        },
        'mistral_7b': {
            'display_name': 'Mistral-7B-Instruct',
            'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
            'use_qlora': True,
            'quantization': '4bit',
            'batch_size': 2,
            'gradient_accumulation': 8
        },
        'qwen2.5_7b': {
            'display_name': 'Qwen2.5-7B-Instruct',
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',
            'use_qlora': True,
            'quantization': '4bit',
            'batch_size': 2,
            'gradient_accumulation': 8
        },
        'qwen2.5_1.5b': {
            'display_name': 'Qwen2.5-1.5B-Instruct',
            'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
            'use_qlora': False,
            'quantization': 'none',
            'batch_size': 8,
            'gradient_accumulation': 4
        },
        'phi3_mini': {
            'display_name': 'Phi-3-mini',
            'model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'use_qlora': False,
            'quantization': 'none',
            'batch_size': 8,
            'gradient_accumulation': 4
        },
        'xlmr_large': {
            'display_name': 'XLM-RoBERTa-Large',
            'model_name': 'xlm-roberta-large',
            'use_qlora': False,
            'quantization': 'none',
            'batch_size': 8,
            'gradient_accumulation': 4,
            'type': 'Encoder (for generation task)'
        },
        'muril_large': {
            'display_name': 'MuRIL-Large',
            'model_name': 'google/muril-large-cased',
            'use_qlora': False,
            'quantization': 'none',
            'batch_size': 8,
            'gradient_accumulation': 4,
            'type': 'Encoder (for generation task)'
        }
    }
    
    print("Creating model folders...")
    for model_name, model_info in models.items():
        create_model_structure(model_name, model_info)
    
    print("\n✅ All model folders created!")


if __name__ == "__main__":
    main()
