"""
Master script to train all models
Usage: python models/train_all.py [model_name]
If no model_name provided, trains all models sequentially
"""

import os
import sys
import subprocess
import argparse

MODELS = {
    'mt5_large': {
        'name': 'mT5-Large',
        'script': 'models/mt5_large/train.py',
        'gpu_memory': '24GB'
    },
    'xlmr_large': {
        'name': 'XLM-RoBERTa-Large',
        'script': 'models/xlmr_large/train.py',
        'gpu_memory': '12GB'
    },
    'muril_large': {
        'name': 'MuRIL-Large',
        'script': 'models/muril_large/train.py',
        'gpu_memory': '12GB'
    },
    'flan_t5_xl': {
        'name': 'FLAN-T5-XL',
        'script': 'models/flan_t5_xl/train.py',
        'gpu_memory': '24GB'
    }
}

def train_model(model_key):
    """Train a specific model"""
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_key]
    script_path = model_info['script']
    
    if not os.path.exists(script_path):
        print(f"Error: Training script not found: {script_path}")
        return False
    
    print("="*70)
    print(f"Training {model_info['name']}")
    print(f"Script: {script_path}")
    print(f"GPU Memory Required: {model_info['gpu_memory']}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),
            check=True
        )
        print(f"\n✅ {model_info['name']} training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_info['name']} training failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train models for POCSO Legal Dialogue')
    parser.add_argument(
        'model',
        nargs='?',
        choices=list(MODELS.keys()) + ['all'],
        default='all',
        help='Model to train (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        print("="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        print(f"\nModels to train: {len(MODELS)}")
        for key, info in MODELS.items():
            print(f"  - {info['name']} ({key})")
        
        print("\nStarting training...")
        results = {}
        for model_key in MODELS.keys():
            results[model_key] = train_model(model_key)
            print("\n")
        
        print("="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        for model_key, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{MODELS[model_key]['name']:25}: {status}")
    else:
        train_model(args.model)

if __name__ == '__main__':
    main()
