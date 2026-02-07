#!/bin/bash
# Run Exp2 pretraining for all 5 generation models on legal corpus (all_cases.txt).
# One model at a time on one GPU to avoid OOM. Run from repo root.
#
# Usage:
#   ./START_EXP2_PRETRAINING.sh
#   CUDA_VISIBLE_DEVICES=4 ./START_EXP2_PRETRAINING.sh   # use GPU 4

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CORPUS_DIR="experiments/exp2_pretraining_only/pretraining/legal_corpus"
if [ ! -f "$CORPUS_DIR/all_cases.txt" ]; then
  echo "Legal corpus not found. Run: python data/prepare_legal_corpus.py --use-all-cases"
  exit 1
fi

MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
LOG_DIR="models/exp2_pretraining_logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Exp2 Pretraining (legal corpus)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

for model in "${MODELS[@]}"; do
  echo ""
  echo "=============================================="
  echo "Pretraining: $model"
  echo "=============================================="
  logfile="$LOG_DIR/pretrain_${model}.log"
  if python3 models/pretrain_template.py --model "$model" --experiment exp2 2>&1 | tee "$logfile"; then
    echo "✅ $model done."
  else
    echo "❌ $model failed. Check $logfile"
  fi
done

echo ""
echo "=============================================="
echo "Exp2 pretraining run finished."
echo "=============================================="
