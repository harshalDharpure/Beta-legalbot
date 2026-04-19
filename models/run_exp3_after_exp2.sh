#!/bin/bash
# After Exp2 completes on GPU 0: run Exp3 then update table. Pass Exp2 PID as first arg.
set -e
cd /DATA/vaneet_2221cs15/legal-bot
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXP2_PID="${1:-}"

if [[ -n "$EXP2_PID" ]]; then
  echo "Waiting for Exp2 process $EXP2_PID to finish (checking every 2 min)..."
  while kill -0 "$EXP2_PID" 2>/dev/null; do sleep 120; done
  echo "Exp2 process finished."
fi

echo "=== Exp3 (pretrain + finetune + eval) for new models on GPU 0 ==="
python3 models/run_new_models_exp123.py --exp 3 --no-update-table

echo "=== Updating category_wise_results table ==="
python3 models/evaluation_results/update_all_results_in_table.py

echo "=== Done ==="
