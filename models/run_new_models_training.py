#!/usr/bin/env python3
"""
Run training for all NEW models only (Gemma 3, Qwen 3, Llama 4 Scout).
Use when you have free GPUs and have accepted any gated licenses on Hugging Face.

Prerequisites:
  - Free GPU(s): run   nvidia-smi   and ensure at least one GPU has ~20GB+ free.
  - Gated models: accept licenses at:
      https://huggingface.co/google/gemma-3-4b-it (and 12b, 27b)
      https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
    Then:   huggingface-cli login
  - Qwen3-8B is open (no login required).

Usage (from repo root):
  python models/run_new_models_training.py

This will train each new model that does not already have an Exp1 checkpoint,
one after another on GPU 0. To use a different GPU: CUDA_VISIBLE_DEVICES=1 python models/run_new_models_training.py
"""

import os
import sys
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(REPO_ROOT)

NEW_MODELS = ['qwen3_8b', 'gemma3_4b', 'gemma3_12b']
SCRIPT = os.path.join('models', 'train_generation_template.py')


def has_exp1_checkpoint(model_name):
    p = os.path.join('models', model_name, 'checkpoints', 'exp1', 'final')
    if not os.path.isdir(p):
        return False
    # Full save has config.json; PEFT save has adapter_config.json
    return os.path.exists(os.path.join(p, 'config.json')) or os.path.exists(os.path.join(p, 'adapter_config.json'))


def main():
    # Use first visible GPU (0). Set CUDA_VISIBLE_DEVICES=1 to use physical GPU 1.
    gpu_id = '0'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print('='*60)
    print('Training NEW models (Exp1)')
    print('CUDA_VISIBLE_DEVICES:', visible or '(all)')
    print('Using GPU index:', gpu_id)
    print('='*60)
    for model in NEW_MODELS:
        if has_exp1_checkpoint(model):
            print(f'\n[SKIP] {model} already has Exp1 checkpoint')
            continue
        print(f'\n[RUN] {model}')
        cmd = [sys.executable, SCRIPT, '--model', model, '--experiment', 'exp1', '--gpu', gpu_id]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'[FAIL] {model} exited with {ret.returncode}. Retrying with QLoRA (--force-qlora)...')
            cmd_qlora = cmd + ['--force-qlora']
            ret2 = subprocess.run(cmd_qlora)
            if ret2.returncode != 0:
                print(f'[FAIL] {model} still failed with QLoRA. Fix and re-run.')
                sys.exit(ret2.returncode)
            print(f'[OK] {model} completed with QLoRA.')
    print('\n[DONE] All new models trained (or skipped).')


if __name__ == '__main__':
    main()
