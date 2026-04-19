#!/usr/bin/env python3
"""
Run Exp4 evaluation for the ORIGINAL 5 models only (5 × 3 configs = 15 runs)
to populate NLI scores in Exp4 result JSONs. Use this once to fill missing Exp4 NLI.

Usage (from repo root):
  python models/run_exp4_nli_original_models.py
"""

import os
import sys
import subprocess

ORIGINAL_MODELS = [
    'llama3.1_8b', 'mistral_7b', 'qwen2.5_7b', 'qwen2.5_1.5b', 'phi3_mini',
]
EXP4_CONFIGS = [
    'hindi_code_mixed_to_english',
    'english_code_mixed_to_hindi',
    'hindi_english_to_code_mixed',
]


def main():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(models_dir)
    os.chdir(repo_root)
    script = os.path.join(models_dir, 'evaluate_generation.py')

    runs = [{'model': m, 'config': c} for m in ORIGINAL_MODELS for c in EXP4_CONFIGS]
    print(f"Exp4 NLI for original 5 models: {len(runs)} runs")

    for i, r in enumerate(runs):
        cmd = [
            sys.executable, script,
            '--model', r['model'],
            '--experiment', 'exp4',
            '--config', r['config'],
        ]
        print(f"\n[{i+1}/{len(runs)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    print("\nDone. Exp4 result JSONs for original models now include NLI.")


if __name__ == '__main__':
    main()
