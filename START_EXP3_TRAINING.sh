#!/bin/bash
# Exp3: Finetune all 5 models from Exp2 checkpoints on dialogue data.
# Each model saves to models/{model}/checkpoints/exp3/final.
# Use one GPU per run to avoid OOM (or set GPUS for parallel runs).
#
# Usage:
#   ./START_EXP3_TRAINING.sh                    # sequential on GPU 0
#   CUDA_VISIBLE_DEVICES=1 ./START_EXP3_TRAINING.sh
#   ./START_EXP3_TRAINING.sh qwen2.5_1.5b       # run only one model

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
LOG_DIR="models/exp3_training_logs"
mkdir -p "$LOG_DIR"

# Optional: run only specified model(s)
if [ -n "$1" ]; then
  MODELS=("$@")
fi

echo "=============================================="
echo "Exp3: Finetuning from Exp2 on dialogue data"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Models: ${MODELS[*]}"
echo "=============================================="

for model in "${MODELS[@]}"; do
  log="$LOG_DIR/exp3_${model}.log"
  echo "[$(date +%H:%M:%S)] Starting $model -> $log"
  python3 models/train_generation_template.py --model "$model" --experiment exp3 --gpu 0 \
    2>&1 | tee "$log" || true
  echo "[$(date +%H:%M:%S)] Finished $model"
done

echo ""
echo "=============================================="
echo "Exp3 training complete. Checkpoints:"
echo "  models/{model}/checkpoints/exp3/final"
echo "Run evaluation: python3 models/evaluate_generation.py --model <model> --experiment exp3"
echo "=============================================="
