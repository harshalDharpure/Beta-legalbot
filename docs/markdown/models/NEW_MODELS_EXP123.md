# Running Exp1, Exp2, Exp3 for New Models (Qwen3-8B, Gemma-3-4B, Gemma-3-12B)

## Overview

- **Exp1**: Fine-tuning only. Run **evaluation** only for models that already have `checkpoints/exp1/final` but lack `results/exp1_results.json`.
- **Exp2**: Pretrain on legal corpus → evaluate zero-shot. Produces `checkpoints/exp2/pretrained/final` and `results/exp2_results.json`.
- **Exp3**: Pretrain → fine-tune on dialogue → evaluate. Produces `checkpoints/exp3/pretrained/final`, `checkpoints/exp3/final`, and `results/exp3_results.json`.

## Prerequisites

- Legal corpus: `experiments/exp2_pretraining_only/pretraining/legal_corpus/` and `experiments/exp3_pretraining_finetuning/pretraining/legal_corpus/` (with `.txt` files).
- GPU: set `CUDA_VISIBLE_DEVICES=0` (or another free GPU). Exp2/Exp3 are long-running (hours per model).

## Commands (from repo root)

### Exp1 only (evaluation; fill results where checkpoint exists)

```bash
python models/run_new_models_exp123.py --exp 1
```

Skips models that already have `results/exp1_results.json`. Updates the category-wise results table at the end unless you pass `--no-update-table`.

### Exp2 only (pretrain + eval, per model)

```bash
python models/run_new_models_exp123.py --exp 2
```

Use a free GPU, e.g. `CUDA_VISIBLE_DEVICES=1 python models/run_new_models_exp123.py --exp 2`.

### Exp3 only (pretrain + finetune + eval, per model)

```bash
python models/run_new_models_exp123.py --exp 3
```

### All experiments (Exp1 eval, then Exp2, then Exp3)

```bash
python models/run_new_models_exp123.py --exp all
```

### Single model

```bash
python models/run_new_models_exp123.py --exp 2 --model gemma3_4b
python models/run_new_models_exp123.py --exp 3 --model qwen3_8b --gpu 1
```

### Background / nohup (for long Exp2/Exp3)

```bash
nohup python models/run_new_models_exp123.py --exp 2 --model qwen3_8b > logs/exp2_qwen3_8b.log 2>&1 &
nohup python models/run_new_models_exp123.py --exp 3 --model qwen3_8b > logs/exp3_qwen3_8b.log 2>&1 &
```

## After new result files appear

Regenerate the markdown tables from result JSONs:

```bash
python models/evaluation_results/update_all_results_in_table.py
```

This updates `models/evaluation_results/category_wise_results_all_experiments.md` with all available Exp1/Exp2/Exp3 metrics for every model (including the three new ones).

## Manual steps (if you prefer not to use the runner)

**Exp1 evaluation only:**

```bash
python models/evaluate_generation.py --model gemma3_4b --experiment exp1
python models/evaluate_generation.py --model gemma3_12b --experiment exp1
```

**Exp2 (pretrain then eval):**

```bash
python models/pretrain_template.py --model qwen3_8b --experiment exp2 --gpu 0
python models/evaluate_generation.py --model qwen3_8b --experiment exp2
```

**Exp3 (pretrain, finetune, eval):**

```bash
python models/pretrain_template.py --model qwen3_8b --experiment exp3 --gpu 0
python models/train_generation_template.py --model qwen3_8b --experiment exp3 --gpu 0
python models/evaluate_generation.py --model qwen3_8b --experiment exp3
```

Replace `qwen3_8b` with `gemma3_4b` or `gemma3_12b` as needed.
