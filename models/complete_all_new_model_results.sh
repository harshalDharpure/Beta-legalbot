#!/bin/bash
# Complete all missing results for new models on one free GPU.
# Runs: gemma3_12b Exp1 (rerun for valid metrics), gemma3_4b Exp2/3, gemma3_12b Exp2/3, then update table.
set -e
cd /DATA/vaneet_2221cs15/legal-bot
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null | head -1

echo "=== 1. Rerun Gemma-3-12B Exp1 (get valid metrics; current has 0 candidate length) ==="
python3 models/evaluate_generation.py --model gemma3_12b --experiment exp1

echo "=== 2. Gemma-3-4B Exp2 (pretrain + eval) ==="
python3 models/run_new_models_exp123.py --exp 2 --model gemma3_4b --no-update-table

echo "=== 3. Gemma-3-12B Exp2 (pretrain + eval) ==="
python3 models/run_new_models_exp123.py --exp 2 --model gemma3_12b --no-update-table

echo "=== 4. Gemma-3-4B Exp3 (pretrain + finetune + eval) ==="
python3 models/run_new_models_exp123.py --exp 3 --model gemma3_4b --no-update-table

echo "=== 5. Gemma-3-12B Exp3 (pretrain + finetune + eval) ==="
python3 models/run_new_models_exp123.py --exp 3 --model gemma3_12b --no-update-table

echo "=== 6. Update category_wise_results table ==="
python3 models/evaluation_results/update_all_results_in_table.py

echo "=== Done: all results complete ==="
