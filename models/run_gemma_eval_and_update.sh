#!/usr/bin/env bash
# Run Gemma evaluations (Exp1/2/3 for gemma3_4b and gemma3_12b) then update category_wise_results.
# Use after gemma3_4b exp1 is done, or run all 6 from scratch with GPU free.
# Usage: CUDA_VISIBLE_DEVICES=0 bash models/run_gemma_eval_and_update.sh

set -e
cd "$(dirname "$0")/.."
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== gemma3_4b exp1 ==="
python3 models/evaluate_generation.py --model gemma3_4b --experiment exp1
echo "=== gemma3_4b exp2 ==="
python3 models/evaluate_generation.py --model gemma3_4b --experiment exp2
echo "=== gemma3_4b exp3 ==="
python3 models/evaluate_generation.py --model gemma3_4b --experiment exp3
echo "=== gemma3_12b exp1 ==="
python3 models/evaluate_generation.py --model gemma3_12b --experiment exp1
echo "=== gemma3_12b exp2 ==="
python3 models/evaluate_generation.py --model gemma3_12b --experiment exp2
echo "=== gemma3_12b exp3 ==="
python3 models/evaluate_generation.py --model gemma3_12b --experiment exp3
echo "=== Updating category_wise_results_all_experiments.md ==="
python3 models/evaluation_results/update_all_results_in_table.py
echo "Done."
