#!/usr/bin/env bash
# Run all experiments for XLM-RoBERTa-Large: Exp1, Exp2, Exp3.
# Exp1: Fine-tuning only (classification on dialogue).
# Exp2: MLM pretraining + train classification head only (frozen encoder), then eval.
# Exp3: MLM pretraining + fine-tuning (classification).
#
# Usage: from repo root:
#   GPU=0 ./RUN_XLMROBERTA_ALL_EXP.sh
#   Or: cd models/xlmr_large && python train.py --experiment exp1 && ...
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
XLMR="models/xlmr_large"
LOG="models/xlmr_large/logs/run_all_exp.log"
mkdir -p "$(dirname "$LOG")"

PY="${PY:-python3}"
echo "========== XLM-RoBERTa Exp1: Fine-tuning only ==========" | tee -a "$LOG"
$PY "$XLMR/train.py" --experiment exp1 2>&1 | tee -a "$LOG"
$PY "$XLMR/evaluate_and_save.py" --experiment exp1 2>&1 | tee -a "$LOG"

echo "========== XLM-RoBERTa Exp2: Pretraining (MLM) + train head only ==========" | tee -a "$LOG"
$PY "$XLMR/pretrain_mlm.py" --experiment exp2 2>&1 | tee -a "$LOG"
$PY "$XLMR/train.py" --experiment exp2 2>&1 | tee -a "$LOG"
$PY "$XLMR/evaluate_and_save.py" --experiment exp2 2>&1 | tee -a "$LOG"

echo "========== XLM-RoBERTa Exp3: Pretraining + Fine-tuning ==========" | tee -a "$LOG"
$PY "$XLMR/pretrain_mlm.py" --experiment exp3 2>&1 | tee -a "$LOG"
$PY "$XLMR/train.py" --experiment exp3 2>&1 | tee -a "$LOG"
$PY "$XLMR/evaluate_and_save.py" --experiment exp3 2>&1 | tee -a "$LOG"

echo "========== All XLM-RoBERTa experiments finished ==========" | tee -a "$LOG"
