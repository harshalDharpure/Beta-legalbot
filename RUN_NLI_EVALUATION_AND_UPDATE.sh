#!/usr/bin/env bash
# Run evaluate_generation.py for Exp1, Exp2, Exp3 for all 5 LLMs to populate nli_score,
# then update category_wise_results_all_experiments.md with NLI values.
set -e
cd "$(dirname "$0")"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
PY="${PY:-python3}"
LOG="models/evaluation_results/nli_eval.log"
mkdir -p models/evaluation_results

echo "========== NLI evaluation (Exp1, Exp2, Exp3 for all generation models) ==========" | tee -a "$LOG"
for model in llama3.1_8b mistral_7b qwen2.5_7b qwen2.5_1.5b phi3_mini; do
  for exp in exp1 exp2 exp3; do
    echo "[$(date)] $model $exp" | tee -a "$LOG"
    $PY models/evaluate_generation.py --model "$model" --experiment "$exp" >> "$LOG" 2>&1 || true
  done
done
echo "========== Updating NLI in category_wise_results table ==========" | tee -a "$LOG"
$PY models/evaluation_results/update_nli_in_table.py >> "$LOG" 2>&1
echo "========== Done. Check $LOG and models/evaluation_results/category_wise_results_all_experiments.md ==========" | tee -a "$LOG"
