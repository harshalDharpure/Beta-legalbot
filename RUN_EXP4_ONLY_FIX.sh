#!/bin/bash
# Re-run Exp4 only (15 jobs) with fixed checkpoint naming. Use after trainer fix for exp4_config.
# Usage: GPUS="0 1 2 3 4" nohup ./RUN_EXP4_ONLY_FIX.sh > models/exp4_exp5_logs/exp4_fix.log 2>&1 &
set -e
cd "$(dirname "$0")"
GPUS="${GPUS:-0 1 2 3 4}"
LOG_DIR="models/exp4_exp5_logs"
JOBS_DIR="$LOG_DIR/jobs"
mkdir -p "$LOG_DIR" "$JOBS_DIR"
export EXP4_EXP5_EPOCHS=2

JOBLIST="$JOBS_DIR/exp4_only.txt"
> "$JOBLIST"
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
EXP4_CONFIGS=(hindi_code_mixed_to_english english_code_mixed_to_hindi hindi_english_to_code_mixed)
for model in "${MODELS[@]}"; do
  for config in "${EXP4_CONFIGS[@]}"; do
    [ -f "models/$model/results/exp4_${config}_results.json" ] && continue
    echo "exp4 $model $config" >> "$JOBLIST"
  done
done
n=$(wc -l < "$JOBLIST")
echo "Exp4 fix: $n jobs to run (skip if results already exist)"

gpu_array=($GPUS)
num_gpus=${#gpu_array[@]}
for ((i=0; i<num_gpus; i++)); do > "$JOBS_DIR/gpu_${i}.txt"; done
idx=0
while IFS= read -r line; do
  echo "$line" >> "$JOBS_DIR/gpu_$((idx % num_gpus)).txt"
  ((idx++)) || true
done < "$JOBLIST"

run_one() {
  local gpu_id=$1
  local line=$2
  read -r typ model rest <<< "$line"
  export CUDA_VISIBLE_DEVICES=$gpu_id
  echo "[GPU$gpu_id] $(date) Exp4 $model $rest"
  python3 models/train_generation_template.py --model "$model" --experiment exp4 --config "$rest" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${rest}.log" || true
  python3 models/evaluate_generation.py --model "$model" --experiment exp4 --config "$rest" 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${rest}.log" || true
  echo "[GPU$gpu_id] Done: $line"
}

worker() {
  local idx=$1 gpu_id=$2
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    run_one "$gpu_id" "$line"
  done < "$JOBS_DIR/gpu_${idx}.txt"
  echo "[GPU$gpu_id] Worker finished $(date)"
}

echo "========== EXP4 FIX RUN started $(date) =========="
for ((i=0; i<num_gpus; i++)); do worker "$i" "${gpu_array[i]}" & done
wait
echo "========== EXP4 FIX RUN finished $(date) =========="
