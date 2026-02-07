"""
Batch evaluation script for all Exp1 models.
Evaluates all 5 generation models on Exp1 test set.
"""

import os
import subprocess
import sys

models = [
    'llama3.1_8b',
    'mistral_7b',
    'qwen2.5_7b',
    'qwen2.5_1.5b',
    'phi3_mini'
]

def evaluate_all():
    """Evaluate all Exp1 models."""
    print("="*70)
    print("Evaluating All Exp1 Models")
    print("="*70)
    print(f"Models to evaluate: {len(models)}")
    print()
    
    results = []
    
    for model in models:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model}")
        print(f"{'='*70}")
        
        checkpoint_path = os.path.join('models', model, 'checkpoints', 'exp1', 'final')
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        
        try:
            # Run evaluation
            cmd = [
                sys.executable,
                'models/evaluate_generation.py',
                '--model', model,
                '--experiment', 'exp1'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {model} evaluation completed")
                results.append((model, 'success'))
            else:
                print(f"❌ {model} evaluation failed")
                print(result.stderr)
                results.append((model, 'failed'))
                
        except Exception as e:
            print(f"❌ Error evaluating {model}: {e}")
            results.append((model, 'error'))
    
    # Summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    for model, status in results:
        print(f"{model}: {status}")
    
    success_count = sum(1 for _, status in results if status == 'success')
    print(f"\n✅ Successfully evaluated: {success_count}/{len(models)} models")


if __name__ == '__main__':
    evaluate_all()
