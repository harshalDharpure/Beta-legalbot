#!/bin/bash
# Run remaining experiments in parallel: Exp2 on two GPUs, then Exp3 on two GPUs when Exp2 done, then update table.
set -e
cd /DATA/vaneet_2221cs15/legal-bot

# Exp2 in parallel: gemma3_4b on GPU 4, gemma3_12b on GPU 1
echo "=== Launching Exp2 in parallel (gemma3_4b on GPU 4, gemma3_12b on GPU 1) ==="
CUDA_VISIBLE_DEVICES=4 python3 models/run_new_models_exp123.py --exp 2 --model gemma3_4b --no-update-table &
PID4=$!
CUDA_VISIBLE_DEVICES=1 python3 models/run_new_models_exp123.py --exp 2 --model gemma3_12b --no-update-table &
PID1=$!
echo "Exp2 gemma3_4b PID=$PID4 (GPU 4), Exp2 gemma3_12b PID=$PID1 (GPU 1)"
wait $PID4 && echo "Exp2 gemma3_4b done." || true
wait $PID1 && echo "Exp2 gemma3_12b done." || true

# Exp3 in parallel: same GPUs
echo "=== Launching Exp3 in parallel (gemma3_4b on GPU 4, gemma3_12b on GPU 1) ==="
CUDA_VISIBLE_DEVICES=4 python3 models/run_new_models_exp123.py --exp 3 --model gemma3_4b --no-update-table &
PID4=$!
CUDA_VISIBLE_DEVICES=1 python3 models/run_new_models_exp123.py --exp 3 --model gemma3_12b --no-update-table &
PID1=$!
echo "Exp3 gemma3_4b PID=$PID4 (GPU 4), Exp3 gemma3_12b PID=$PID1 (GPU 1)"
wait $PID4 && echo "Exp3 gemma3_4b done." || true
wait $PID1 && echo "Exp3 gemma3_12b done." || true

echo "=== Updating category_wise_results table ==="
python3 models/evaluation_results/update_all_results_in_table.py
echo "=== Done ==="
