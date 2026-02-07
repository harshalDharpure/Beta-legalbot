"""
Setup experiment folders and copy data splits.
"""

import os
import shutil
from pathlib import Path


def setup_experiment_folders():
    """Create experiment folder structure."""
    
    experiments = {
        'exp1_finetuning_only': {
            'description': 'Exp1: Finetuning only (baseline)',
            'files': ['train_70.jsonl', 'val_10.jsonl', 'test_20.jsonl']
        },
        'exp2_pretraining_only': {
            'description': 'Exp2: Pretraining only (zero-shot evaluation)',
            'files': ['test_20.jsonl']  # Only test for zero-shot evaluation
        },
        'exp3_pretraining_finetuning': {
            'description': 'Exp3: Pretraining + Finetuning (full pipeline)',
            'files': ['train_70.jsonl', 'val_10.jsonl', 'test_20.jsonl']
        },
        'exp4_zeroshot_transfer': {
            'description': 'Exp4: Zero-shot cross-lingual transfer',
            'files': []  # Will be created separately
        },
        'exp5_fewshot_learning': {
            'description': 'Exp5: Few-shot learning',
            'files': []  # Will be created separately
        }
    }
    
    base_dir = Path('experiments')
    splits_dir = Path('data/splits')
    
    # Create experiment folders
    for exp_name, exp_info in experiments.items():
        exp_dir = base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme_path = exp_dir / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {exp_info['description']}\n\n")
            f.write(f"{exp_info['description']}\n\n")
            f.write("## Files\n\n")
            for file in exp_info['files']:
                f.write(f"- `{file}`\n")
        
        # Copy data files
        data_dir = exp_dir / 'data'
        data_dir.mkdir(exist_ok=True)
        
        for file in exp_info['files']:
            src = splits_dir / file
            if src.exists():
                dst = data_dir / file
                shutil.copy2(src, dst)
                print(f"✅ Copied {file} to {exp_dir}/data/")
    
    # Create exp2 pretraining folder
    exp2_pretrain_dir = base_dir / 'exp2_pretraining_only' / 'pretraining'
    exp2_pretrain_dir.mkdir(parents=True, exist_ok=True)
    
    # Create exp2 evaluation folder
    exp2_eval_dir = base_dir / 'exp2_pretraining_only' / 'evaluation'
    exp2_eval_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(splits_dir / 'test_20.jsonl', exp2_eval_dir / 'test.jsonl')
    
    # Create exp3 pretraining and finetuning folders
    exp3_pretrain_dir = base_dir / 'exp3_pretraining_finetuning' / 'pretraining'
    exp3_pretrain_dir.mkdir(parents=True, exist_ok=True)
    
    exp3_finetune_dir = base_dir / 'exp3_pretraining_finetuning' / 'finetuning'
    exp3_finetune_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy data to exp3 finetuning
    for file in ['train_70.jsonl', 'val_10.jsonl', 'test_20.jsonl']:
        src = splits_dir / file
        if src.exists():
            dst = exp3_finetune_dir / file.replace('_70', '').replace('_10', '').replace('_20', '')
            shutil.copy2(src, dst)
            print(f"✅ Copied {file} to {exp3_finetune_dir}/")
    
    print("\n✅ Experiment folders created!")


if __name__ == "__main__":
    setup_experiment_folders()
