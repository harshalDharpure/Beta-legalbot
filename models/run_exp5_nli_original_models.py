#!/usr/bin/env python3
"""
Run Exp5 evaluation for the ORIGINAL 5 models only to populate NLI scores
in Exp5 result JSONs. Runs 5 models × 4 few-sizes × 2 directions = 40 runs.

Usage (from repo root):
  python models/run_exp5_nli_original_models.py
"""

import os
import sys
import subprocess

ORIGINAL_MODELS = [
    'llama3.1_8b', 'mistral_7b', 'qwen2.5_7b', 'qwen2.5_1.5b', 'phi3_mini',
]
EXP5_FEW_SIZES = [5, 10, 20, 50]
EXP5_DIRECTIONS = [
    'hindi_code_mixed_to_english',
    'english_code_mixed_to_hindi',
]


def main():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(models_dir)
    os.chdir(repo_root)
    script = os.path.join(models_dir, 'evaluate_generation.py')

    runs = [
        {'model': m, 'few_size': f, 'direction': d}
        for m in ORIGINAL_MODELS
        for f in EXP5_FEW_SIZES
        for d in EXP5_DIRECTIONS
    ]
    print(f"Exp5 NLI for original 5 models: {len(runs)} runs")

    for i, r in enumerate(runs):
        cmd = [
            sys.executable, script,
            '--model', r['model'],
            '--experiment', 'exp5',
            '--few-size', str(r['few_size']),
            '--direction', r['direction'],
        ]
        print(f"\n[{i+1}/{len(runs)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    print("\nDone. Exp5 result JSONs for original models now include NLI.")


if __name__ == '__main__':
    main()
