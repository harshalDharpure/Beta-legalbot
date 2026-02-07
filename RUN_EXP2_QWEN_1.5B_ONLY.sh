#!/bin/bash
# Exp2 pretraining for Qwen2.5-1.5B only.
# This model uses full fine-tuning (no QLoRA) and needs a dedicated GPU (~25 GiB).
# Use a GPU that is not shared with other processes (e.g. GPU 1 if GPU 0 is busy).
#
# Usage:
#   ./RUN_EXP2_QWEN_1.5B_ONLY.sh
#   CUDA_VISIBLE_DEVICES=2 ./RUN_EXP2_QWEN_1.5B_ONLY.sh   # use GPU 2

set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

LOG_DIR="models/exp2_pretraining_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pretrain_qwen2.5_1.5b_exp2.log"

echo "Exp2 pretraining: Qwen2.5-1.5B on GPU $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG"
python3 models/pretrain_template.py --model qwen2.5_1.5b --experiment exp2 2>&1 | tee "$LOG"
echo "Done. Checkpoint: models/qwen2.5_1.5b/checkpoints/exp2/pretrained/final"
