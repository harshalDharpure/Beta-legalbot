#!/usr/bin/env python3
"""
Re-run Exp4 evaluation for all 5 models × 3 configs to populate NLI scores
(and full per-domain metrics if the evaluation script was updated).
Usage: from repo root, run:
  python models/run_exp4_nli.py
Or run a single model/config:
  EXP4_MODEL=llama3.1_8b EXP4_CONFIG=hindi_code_mixed_to_english python models/run_exp4_nli.py
"""

import os
import sys
import subprocess

MODELS = [
    'gemma3_4b', 'gemma3_12b', 'qwen3_8b',
]
EXP4_CONFIGS = [
    'hindi_code_mixed_to_english',
    'english_code_mixed_to_hindi',
    'hindi_english_to_code_mixed',
]


def main():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(models_dir)  # repo root = directory containing 'models'
    os.chdir(repo_root)
    script = os.path.join(models_dir, 'evaluate_generation.py')

    model_arg = os.environ.get('EXP4_MODEL')
    config_arg = os.environ.get('EXP4_CONFIG')

    if model_arg and config_arg:
        runs = [{'model': model_arg, 'config': config_arg}]
    else:
        runs = [{'model': m, 'config': c} for m in MODELS for c in EXP4_CONFIGS]

    for i, r in enumerate(runs):
        cmd = [
            sys.executable, script,
            '--model', r['model'],
            '--experiment', 'exp4',
            '--config', r['config'],
        ]
        print(f"\n[{i+1}/{len(runs)}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    print("\nDone. Exp4 result JSONs now include NLI (and full per-domain metrics if script was updated).")


if __name__ == '__main__':
    main()
