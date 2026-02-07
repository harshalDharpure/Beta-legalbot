#!/bin/bash
# Script to evaluate all Exp1 models (one at a time to avoid GPU OOM).
# Run from repo root: ./START_EXP1_EVALUATION.sh
# Use one GPU so memory is freed between models (set GPU id below).

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=========================================="
echo "Evaluating All Exp1 Models (GPU $CUDA_VISIBLE_DEVICES)"
echo "=========================================="

# Smallest first to reduce OOM risk when running sequentially
models=("qwen2.5_1.5b" "phi3_mini" "qwen2.5_7b" "mistral_7b" "llama3.1_8b")

for model in "${models[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating: $model"
    echo "=========================================="
    python3 models/evaluate_generation.py --model "$model" --experiment exp1 || true
    echo ""
done

echo "=========================================="
echo "Exp1 evaluation run complete."
echo "=========================================="
