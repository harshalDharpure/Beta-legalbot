#!/bin/bash
# Run llama3.1_8b Exp3 evaluation in a way that survives terminal close.
# Usage: GPU=4 ./RUN_EXP3_LLAMA_NOHUP.sh
# Then you can close the terminal; check log for progress.

set -e
cd "$(dirname "$0")"
GPU="${GPU:-4}"
LOG="models/exp3_evaluation_logs/exp3_llama3.1_8b_nohup.log"
mkdir -p "$(dirname "$LOG")"

echo "Starting llama3.1_8b Exp3 eval on GPU $GPU (immune to terminal close)."
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
nohup env CUDA_VISIBLE_DEVICES=$GPU python3 -u models/evaluate_generation.py --model llama3.1_8b --experiment exp3 >> "$LOG" 2>&1 &
echo "PID: $!"
echo "Results will be: models/llama3.1_8b/results/exp3_results.json"
