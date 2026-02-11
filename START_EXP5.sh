#!/bin/bash
# Exp5: Few-shot learning — train on few-shot data, evaluate on test.
# Usage:
#   GPU=0 ./START_EXP5.sh                    # Run one (default: qwen2.5_1.5b, few10, hindi_code_mixed_to_english)
#   GPU=0 MODEL=phi3_mini FEW=5 DIR=english_code_mixed_to_hindi ./START_EXP5.sh
#   GPU=0 ./START_EXP5.sh all                # Run all 5 models × 4 few sizes × 2 directions (sequential)
set -e
cd "$(dirname "$0")"
GPU="${GPU:-0}"
MODEL="${MODEL:-qwen2.5_1.5b}"
FEW="${FEW:-10}"
DIR="${DIR:-hindi_code_mixed_to_english}"
RUN_ALL="${1:-}"

FEW_SIZES=(5 10 20 50)
DIRECTIONS=(hindi_code_mixed_to_english english_code_mixed_to_hindi)
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
LOG_DIR="models/exp5_logs"
mkdir -p "$LOG_DIR"

run_one() {
  local model=$1
  local few=$2
  local dir=$3
  local logfile="$LOG_DIR/exp5_${model}_few${few}_${dir}.log"
  echo "[$(date)] Training $model on exp5 few=$few direction=$dir (GPU $GPU) -> $logfile"
  CUDA_VISIBLE_DEVICES=$GPU python3 models/train_generation_template.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" 2>&1 | tee "$logfile"
  echo "[$(date)] Evaluating $model on exp5 few=$few direction=$dir"
  CUDA_VISIBLE_DEVICES=$GPU python3 models/evaluate_generation.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" 2>&1 | tee -a "$logfile"
  echo "[$(date)] Done $model few$few $dir"
}

if [ "$RUN_ALL" = "all" ]; then
  for m in "${MODELS[@]}"; do
    for few in "${FEW_SIZES[@]}"; do
      for d in "${DIRECTIONS[@]}"; do
        run_one "$m" "$few" "$d"
      done
    done
  done
  echo "Exp5 full run complete. Results: models/*/results/exp5_*_results.json"
else
  run_one "$MODEL" "$FEW" "$DIR"
  echo "Exp5 single run complete. Results: models/$MODEL/results/exp5_few${FEW}_${DIR}_results.json"
fi
