#!/bin/bash
# Run after Exp1 evals complete: Exp2 for all new models, then Exp3, then update table.
# Usage: nohup bash models/run_exp2_exp3_then_update.sh >> logs/exp2_exp3.log 2>&1 &
set -e
cd /DATA/vaneet_2221cs15/legal-bot
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Waiting for Exp1 results (gemma3_4b, gemma3_12b) ==="
for _ in $(seq 1 3600); do
  if [[ -f models/gemma3_4b/results/exp1_results.json && -f models/gemma3_12b/results/exp1_results.json ]]; then
    echo "Exp1 results found. Proceeding to Exp2."
    break
  fi
  sleep 10
done
if [[ ! -f models/gemma3_4b/results/exp1_results.json || ! -f models/gemma3_12b/results/exp1_results.json ]]; then
  echo "Timeout waiting for Exp1 results. Proceeding anyway (Exp2/Exp3 may run for models that have checkpoints)."
fi

echo "=== Exp2 (pretrain + eval) for new models ==="
python3 models/run_new_models_exp123.py --exp 2 --no-update-table

echo "=== Exp3 (pretrain + finetune + eval) for new models ==="
python3 models/run_new_models_exp123.py --exp 3 --no-update-table

echo "=== Updating category_wise_results table ==="
python3 models/evaluation_results/update_all_results_in_table.py

echo "=== Done ==="
