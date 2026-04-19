#!/bin/bash
# Re-run failed jobs on a free GPU: Exp2 for gemma3_4b and gemma3_12b, then Exp1 for gemma3_12b, then update table.
set -e
cd /DATA/vaneet_2221cs15/legal-bot
GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU: $GPU"

echo "=== 1/3 Exp2 gemma3_4b (pretrain + eval) ==="
python3 models/run_new_models_exp123.py --exp 2 --model gemma3_4b --no-update-table

echo "=== 2/3 Exp2 gemma3_12b (pretrain + eval) ==="
python3 models/run_new_models_exp123.py --exp 2 --model gemma3_12b --no-update-table

echo "=== 3/3 Gemma-3-12B Exp1 eval ==="
python3 models/evaluate_generation.py --model gemma3_12b --experiment exp1

echo "=== Updating category_wise_results table ==="
python3 models/evaluation_results/update_all_results_in_table.py

echo "=== Done ==="
