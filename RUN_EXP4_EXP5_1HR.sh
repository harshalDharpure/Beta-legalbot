#!/bin/bash
# Finish all Exp4 + Exp5 in ~1 hour: 50 training steps + 30-sample eval per job, 5 GPUs.
# Results are "quick run" (not full quality). For full results use RUN_EXP4_EXP5_ONE_DAY.sh or multi-GPU full.
# Usage: nohup ./RUN_EXP4_EXP5_1HR.sh > models/exp4_exp5_logs/one_hr.log 2>&1 &
set -e
cd "$(dirname "$0")"
export EXP4_EXP5_MAX_STEPS=50
export EVAL_MAX_SAMPLES=30
GPUS="${GPUS:-0 1 2 3 4}"
LOG_DIR="models/exp4_exp5_logs"
JOBS_DIR="$LOG_DIR/jobs"
mkdir -p "$LOG_DIR" "$JOBS_DIR"

JOBLIST="$JOBS_DIR/all_jobs_1hr.txt"
> "$JOBLIST"
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
EXP4_CONFIGS=(hindi_code_mixed_to_english english_code_mixed_to_hindi hindi_english_to_code_mixed)
FEW_SIZES=(5 10 20 50)
EXP5_DIRECTIONS=(hindi_code_mixed_to_english english_code_mixed_to_hindi)

for model in "${MODELS[@]}"; do
  for config in "${EXP4_CONFIGS[@]}"; do
    echo "exp4 $model $config" >> "$JOBLIST"
  done
  for few in "${FEW_SIZES[@]}"; do
    for dir in "${EXP5_DIRECTIONS[@]}"; do
      echo "exp5 $model $few $dir" >> "$JOBLIST"
    done
  done
done
total_jobs=$(wc -l < "$JOBLIST")
echo "1-HR run: $total_jobs jobs, max_steps=$EXP4_EXP5_MAX_STEPS, eval_samples=$EVAL_MAX_SAMPLES, GPUS=$GPUS"

gpu_array=($GPUS)
num_gpus=${#gpu_array[@]}
for ((i=0; i<num_gpus; i++)); do > "$JOBS_DIR/gpu_${i}.txt"; done
idx=0
while IFS= read -r line; do
  echo "$line" >> "$JOBS_DIR/gpu_$((idx % num_gpus)).txt"
  ((idx++)) || true
done < "$JOBLIST"

run_one_job() {
  local gpu_id=$1
  local line=$2
  read -r typ model rest <<< "$line"
  export CUDA_VISIBLE_DEVICES=$gpu_id
  if [ "$typ" = "exp4" ]; then
    local config=$rest
    [ -f "models/$model/results/exp4_${config}_results.json" ] && { echo "[GPU$gpu_id] SKIP exp4 $model $config"; return 0; }
    echo "[GPU$gpu_id] $(date) Exp4 $model $config"
    python3 models/train_generation_template.py --model "$model" --experiment exp4 --config "$config" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp4 --config "$config" 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
  else
    local few="${rest%% *}" dir="${rest#* }"
    [ -f "models/$model/results/exp5_few${few}_${dir}_results.json" ] && { echo "[GPU$gpu_id] SKIP exp5 $model few$few $dir"; return 0; }
    echo "[GPU$gpu_id] $(date) Exp5 $model few$few $dir"
    python3 models/train_generation_template.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
  fi
  echo "[GPU$gpu_id] Done: $line"
}

worker() {
  local idx=$1 gpu_id=$2
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    run_one_job "$gpu_id" "$line"
  done < "$JOBS_DIR/gpu_${idx}.txt"
  echo "[GPU$gpu_id] Worker finished $(date)"
}

echo "========== 1-HR RUN started $(date) =========="
for ((i=0; i<num_gpus; i++)); do
  worker "$i" "${gpu_array[i]}" &
done
wait
echo "========== 1-HR RUN finished $(date) =========="
