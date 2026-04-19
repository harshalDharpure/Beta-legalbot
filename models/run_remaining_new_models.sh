#!/usr/bin/env bash
# Run Exp1 training (if needed) + Exp1/2/3 eval for new models (qwen3_8b, gemma3_4b, gemma3_12b),
# then update category_wise_results_all_experiments.md.
#
# Usage (from repo root):
#   bash models/run_remaining_new_models.sh
# With specific GPU:
#   CUDA_VISIBLE_DEVICES=1 bash models/run_remaining_new_models.sh

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Step 1: Exp1 training (new models missing checkpoint)"
echo "=============================================="
python3 models/run_new_models_training.py

echo ""
echo "=============================================="
echo "Step 2: Exp1 eval + Exp2 + Exp3 + update table"
echo "=============================================="
python3 models/run_new_models_exp123.py --exp all

echo ""
echo "Done. Check models/*/results/ and models/evaluation_results/category_wise_results_all_experiments.md"
