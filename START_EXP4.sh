#!/bin/bash
# Exp4: Zero-shot transfer - train on source lang(s), evaluate on target.
# Usage: GPU=0 ./START_EXP4.sh
#   MODEL=phi3_mini CONFIG=english_code_mixed_to_hindi ./START_EXP4.sh
set -e
cd "$(dirname "$0")"
GPU="${GPU:-0}"
MODEL="${MODEL:-qwen2.5_1.5b}"
CONFIG="${CONFIG:-hindi_code_mixed_to_english}"
LOG_DIR="models/exp4_logs"
mkdir -p "$LOG_DIR"
logfile="$LOG_DIR/exp4_${MODEL}_${CONFIG}.log"
echo "[$(date)] Training $MODEL exp4 config=$CONFIG (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU python3 models/train_generation_template.py --model "$MODEL" --experiment exp4 --config "$CONFIG" 2>&1 | tee "$logfile"
echo "[$(date)] Evaluating $MODEL exp4 config=$CONFIG"
CUDA_VISIBLE_DEVICES=$GPU python3 models/evaluate_generation.py --model "$MODEL" --experiment exp4 --config "$CONFIG" 2>&1 | tee -a "$logfile"
echo "Done. Results: models/$MODEL/results/exp4_${CONFIG}_results.json"
