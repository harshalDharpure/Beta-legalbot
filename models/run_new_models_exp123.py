#!/usr/bin/env python3
"""
Run Exp1 (eval only), Exp2 (pretrain + eval), and Exp3 (pretrain + finetune + eval)
for the NEW models: qwen3_8b, gemma3_4b, gemma3_12b.
Then update category_wise_results_all_experiments.md with the new result JSONs.

Prerequisites:
  - Exp1: checkpoints/exp1/final must exist (train first via run_new_models_training.py if needed).
  - Exp2/Exp3: legal corpus at experiments/exp2_pretraining_only/pretraining/legal_corpus/
    and experiments/exp3_pretraining_finetuning/pretraining/legal_corpus/
  - GPU: set CUDA_VISIBLE_DEVICES=0 (or another free GPU).

Usage (from repo root):
  # Exp1 only (eval; fill results for models that have checkpoint but no results)
  python models/run_new_models_exp123.py --exp 1

  # Exp2 only (pretrain + eval per model)
  python models/run_new_models_exp123.py --exp 2

  # Exp3 only (pretrain + finetune + eval per model)
  python models/run_new_models_exp123.py --exp 3

  # All experiments for all new models (Exp1 eval, then Exp2, then Exp3)
  python models/run_new_models_exp123.py --exp all

  # Single model
  python models/run_new_models_exp123.py --exp 2 --model gemma3_4b
"""

import os
import sys
import subprocess
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(REPO_ROOT)

NEW_MODELS = ['qwen3_8b', 'gemma3_4b', 'gemma3_12b']


def has_checkpoint(model_name, experiment, subpath='final'):
    if experiment == 'exp1':
        p = os.path.join('models', model_name, 'checkpoints', 'exp1', 'final')
    elif experiment == 'exp2':
        p = os.path.join('models', model_name, 'checkpoints', 'exp2', 'pretrained', 'final')
    elif experiment == 'exp3':
        p = os.path.join('models', model_name, 'checkpoints', 'exp3', 'final')
    else:
        return False
    if not os.path.isdir(p):
        return False
    return os.path.exists(os.path.join(p, 'config.json')) or os.path.exists(os.path.join(p, 'adapter_config.json'))


def has_results(model_name, experiment):
    if experiment == 'exp4' or experiment == 'exp5':
        return False
    p = os.path.join('models', model_name, 'results', f'{experiment}_results.json')
    return os.path.isfile(p)


def run_exp1_eval(model_name, gpu_id='0'):
    """Run evaluation only for Exp1 (no training)."""
    if not has_checkpoint(model_name, 'exp1'):
        print(f'  [SKIP] {model_name}: no Exp1 checkpoint')
        return True
    if has_results(model_name, 'exp1'):
        print(f'  [SKIP] {model_name}: Exp1 results already exist')
        return True
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    cmd = [sys.executable, 'models/evaluate_generation.py', '--model', model_name, '--experiment', 'exp1']
    ret = subprocess.run(cmd, env=env)
    return ret.returncode == 0


def run_exp2(model_name, gpu_id='0'):
    """Pretrain (exp2) then evaluate."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    if not has_results(model_name, 'exp2'):
        print(f'  [PRETRAIN] {model_name} Exp2...')
        ret = subprocess.run(
            [sys.executable, 'models/pretrain_template.py', '--model', model_name, '--experiment', 'exp2', '--gpu', gpu_id],
            env=env
        )
        if ret.returncode != 0:
            print(f'  [FAIL] {model_name} Exp2 pretrain')
            return False
        print(f'  [EVAL] {model_name} Exp2...')
        ret = subprocess.run(
            [sys.executable, 'models/evaluate_generation.py', '--model', model_name, '--experiment', 'exp2'],
            env=env
        )
        if ret.returncode != 0:
            print(f'  [FAIL] {model_name} Exp2 eval')
            return False
    else:
        print(f'  [SKIP] {model_name}: Exp2 results already exist')
    return True


def run_exp3(model_name, gpu_id='0'):
    """Pretrain (exp3) then finetune then evaluate."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    if has_results(model_name, 'exp3'):
        print(f'  [SKIP] {model_name}: Exp3 results already exist')
        return True
    print(f'  [PRETRAIN] {model_name} Exp3...')
    ret = subprocess.run(
        [sys.executable, 'models/pretrain_template.py', '--model', model_name, '--experiment', 'exp3', '--gpu', gpu_id],
        env=env
    )
    if ret.returncode != 0:
        print(f'  [FAIL] {model_name} Exp3 pretrain')
        return False
    print(f'  [FINETUNE] {model_name} Exp3...')
    ret = subprocess.run(
        [sys.executable, 'models/train_generation_template.py', '--model', model_name, '--experiment', 'exp3', '--gpu', gpu_id],
        env=env
    )
    if ret.returncode != 0:
        print(f'  [FAIL] {model_name} Exp3 finetune (try --force-qlora if OOM)')
        ret2 = subprocess.run(
            [sys.executable, 'models/train_generation_template.py', '--model', model_name, '--experiment', 'exp3', '--gpu', gpu_id, '--force-qlora'],
            env=env
        )
        if ret2.returncode != 0:
            return False
    print(f'  [EVAL] {model_name} Exp3...')
    ret = subprocess.run(
        [sys.executable, 'models/evaluate_generation.py', '--model', model_name, '--experiment', 'exp3'],
        env=env
    )
    if ret.returncode != 0:
        print(f'  [FAIL] {model_name} Exp3 eval')
        return False
    return True


def update_table():
    """Refresh category_wise_results from result JSONs."""
    ret = subprocess.run([sys.executable, 'models/evaluation_results/update_all_results_in_table.py'])
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run Exp1/2/3 for new models and update results table')
    parser.add_argument('--exp', type=str, default='1', choices=['1', '2', '3', 'all'],
                        help='Which experiment: 1 (eval only), 2 (pretrain+eval), 3 (pretrain+finetune+eval), all')
    parser.add_argument('--model', type=str, default=None,
                        help='Single model (e.g. gemma3_4b); default all new models')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU id (default: 0 or CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--no-update-table', action='store_true', help='Do not run update_all_results_in_table.py at end')
    args = parser.parse_args()

    models = [args.model] if args.model else NEW_MODELS
    if args.model and args.model not in NEW_MODELS:
        print(f'Unknown model: {args.model}. Allowed: {NEW_MODELS}')
        sys.exit(1)

    gpu_id = args.gpu or os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
    print('='*60)
    print('New models Exp1/2/3 runner')
    print('Models:', models)
    print('Experiment:', args.exp)
    print('GPU:', gpu_id)
    print('='*60)

    ok = True
    if args.exp in ('1', 'all'):
        print('\n--- Exp1 (evaluation only) ---')
        for m in models:
            if not run_exp1_eval(m, gpu_id):
                ok = False
    if args.exp in ('2', 'all'):
        print('\n--- Exp2 (pretrain + eval) ---')
        for m in models:
            if not run_exp2(m, gpu_id):
                ok = False
    if args.exp in ('3', 'all'):
        print('\n--- Exp3 (pretrain + finetune + eval) ---')
        for m in models:
            if not run_exp3(m, gpu_id):
                ok = False

    if not args.no_update_table and ok:
        print('\n--- Updating category_wise_results table ---')
        if not update_table():
            ok = False

    print('\n[DONE]' if ok else '\n[FAIL] Some steps failed.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
