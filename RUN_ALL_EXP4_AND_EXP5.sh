#!/bin/bash
# Run all Exp4 (5 models x 3 configs) then all Exp5 (5 models x 4 few x 2 directions) sequentially.
# Usage: GPU=4 nohup ./RUN_ALL_EXP4_AND_EXP5.sh > models/exp4_exp5_master.log 2>&1 &
set -e
cd "$(dirname "$0")"
GPU="${GPU:-4}"
LOG_DIR="models/exp4_exp5_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "========== RUN_ALL_EXP4_AND_EXP5 started $(date) GPU=$GPU =========="

MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
EXP4_CONFIGS=(hindi_code_mixed_to_english english_code_mixed_to_hindi hindi_english_to_code_mixed)
FEW_SIZES=(5 10 20 50)
EXP5_DIRECTIONS=(hindi_code_mixed_to_english english_code_mixed_to_hindi)

# Exp4: 5 x 3 = 15 runs
echo "---------- EXP4 (zero-shot transfer) ----------"
for model in "${MODELS[@]}"; do
  for config in "${EXP4_CONFIGS[@]}"; do
    out="models/$model/results/exp4_${config}_results.json"
    if [ -f "$out" ]; then echo "[$(date)] SKIP Exp4 $model $config (exists: $out)"; continue; fi
    echo "[$(date)] Exp4 $model $config"
    export CUDA_VISIBLE_DEVICES=$GPU
    python3 models/train_generation_template.py --model "$model" --experiment exp4 --config "$config" --gpu 0 2>&1 | tee "$LOG_DIR/exp4_${model}_${config}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp4 --config "$config" 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
  done
done
echo "---------- EXP4 complete $(date) ----------"

# Exp5: 5 x 4 x 2 = 40 runs
echo "---------- EXP5 (few-shot) ----------"
for model in "${MODELS[@]}"; do
  for few in "${FEW_SIZES[@]}"; do
    for dir in "${EXP5_DIRECTIONS[@]}"; do
      out="models/$model/results/exp5_few${few}_${dir}_results.json"
      if [ -f "$out" ]; then echo "[$(date)] SKIP Exp5 $model few$few $dir (exists: $out)"; continue; fi
      echo "[$(date)] Exp5 $model few$few $dir"
      export CUDA_VISIBLE_DEVICES=$GPU
      python3 models/train_generation_template.py --model "$model" --experiment exp5 --few-size $few --direction "$dir" --gpu 0 2>&1 | tee "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
      python3 models/evaluate_generation.py --model "$model" --experiment exp5 --few-size $few --direction "$dir" 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
    done
  done
done
echo "---------- EXP5 complete $(date) ----------"
echo "========== RUN_ALL_EXP4_AND_EXP5 finished $(date) =========="
